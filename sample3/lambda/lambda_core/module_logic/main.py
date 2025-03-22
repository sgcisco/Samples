import time
from datetime import datetime
from logging import Logger
from typing import Any, Dict, Tuple

import pandas as pd
from lambda_core.tools.utils import is_integration_test
from lambda_core.definitions.constants import (
    INFERENCE_INTERVAL,
    INFERENCE_WINDOW,
)
from lambda_core.module_logic.etl import create_dataset
from lambda_core.module_logic.inputs import (
    clean_lambda_caches,
    read_inputs,
    read_watermark,
    write_context_to_cache,
    write_detections_to_rds,
    write_to_lambda_cache,
    write_watermark,
    prepare_detections_for_export,
    prepare_lambda_metrics,
    compute_out_of_vocab_metrics,
)
from lambda_core.module_logic.logical_class import with_logical_class
from lambda_core.module_logic.ml.clf_dataset import ClassificationDataset
from lambda_core.module_logic.ml.features import build_app_features
from lambda_core.module_logic.ml.loader import get_active_lc_definitions, load_ml_models


def main_logic(
    data: Dict[str, Any],
    logger: Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    lc_definitions, model_bundle = load_ml_models(logger)
    active_lc_definitions = get_active_lc_definitions(lc_definitions, model_bundle)

    end_time = datetime.fromtimestamp(data["endTime"])
    customer_id = data["customerId"]
    logger.info(f"Customer id is {customer_id}")
    active_logical_classes = list(model_bundle.keys())

    cur_watermark_ts = read_watermark(customer_id, end_time, logger)

    logger.info(f"Current watermark is {cur_watermark_ts}")
    # Lookback maximum from starting cur_watermark_ts or INFERENCE_INTERVAL is assumed
    start_time = min(
        end_time - INFERENCE_INTERVAL,
        max(cur_watermark_ts, end_time - 4 * INFERENCE_INTERVAL),
    )
    logger.info(f"Start time is {start_time}")
    logger.info(f"End time is {end_time}")

    begin = time.perf_counter()
    nf_pd, endpoint_pd, etl_pd, new_watermark_ts = read_inputs(
        customer_id, start_time, end_time, cur_watermark_ts, logger
    )
    logger.info(f"New watermark {new_watermark_ts}")
    logger.info(f"Files loading took {time.perf_counter() - begin:0.4f} seconds")

    if new_watermark_ts >= end_time:
        logger.info(
            f"New watermark {new_watermark_ts.isoformat()} is greater or equal to end time "
            f"{end_time.isoformat()}, nothing to do"
        )
        return None, None

    write_watermark(new_watermark_ts, customer_id, logger)
    clean_lambda_caches(customer_id, new_watermark_ts, logger)

    if endpoint_pd.empty:
        logger.warning("No Endpoint mappings available")
        return None, None

    begin = time.perf_counter()
    logger.info(f"Endpoint data before logical class assignment: {len(endpoint_pd)} records")
    endpoint_pd_with_class = with_logical_class(
        endpoint_pd, active_lc_definitions, start_time, logger
    )
    logger.info(
        f"Logical class assignment took {time.perf_counter() - begin:0.4f} seconds"
    )

    write_context_to_cache(endpoint_pd_with_class, customer_id, new_watermark_ts, logger)
    logger.info(
        f"Endpoint data after logical class assignment: {len(endpoint_pd_with_class)} records"
    )

    if nf_pd.empty:
        logger.warning(
            "No Netflow data to process. Dataset could not be created. Stopping."
        )
        return None, None

    if endpoint_pd_with_class.empty:
        logger.warning(
            "No Endpoint data to process. Dataset could not be created. Stopping."
        )
        return None, None

    begin = time.perf_counter()

    df_apps, etl_pd_new_data = create_dataset(
        nf_pd,
        endpoint_pd_with_class,
        etl_pd,
        active_logical_classes,
        INFERENCE_INTERVAL,
        INFERENCE_WINDOW,
        logger,
    )

    logger.info(f"ETL creation time: {time.perf_counter() - begin:0.4f} seconds")

    if df_apps.empty:
        logger.warning(
            "No data to process after filtering (scan, MAC and class). "
            "Dataset could not be created. Stopping."
        )
        return None, None

    detections_pds = []
    out_of_vocab_metrics_list = []
    for active_logical_class in active_logical_classes:
        df_apps_active_logical_class = df_apps[df_apps["label"] == active_logical_class]
        if not df_apps_active_logical_class.empty:
            vocabulary = model_bundle[active_logical_class].vocabulary
            begin = time.perf_counter()
            X = build_app_features(
                df_apps_active_logical_class,
                vocabulary,
            )
            logger.info(
                f"Feature construction for {active_logical_class} took"
                f" {time.perf_counter() - begin:0.4f} seconds"
            )
            # TODO: fix vocabularies param and approach to handle it
            input_dataset = ClassificationDataset(
                X=X.toarray(),
                vocabularies={"all": vocabulary},
                df_apps=df_apps_active_logical_class,
            )
            begin = time.perf_counter()
            predictions = model_bundle[active_logical_class].predict(input_dataset)
            logger.info(f"Predictions for {active_logical_class}: {predictions}")
            detections_pds.append(
                df_apps_active_logical_class[
                    ["mac", "label", "windowStart", "windowEnd", "apps", "uuid"]
                ].iloc[predictions.astype(bool)]
            )
            out_of_vocab_metrics_list.append(
                compute_out_of_vocab_metrics(
                    input_dataset, predictions, active_logical_class
                )
            )
            logger.info(
                f"Model evaluation for {active_logical_class} took"
                f" {time.perf_counter() - begin:0.4f} seconds"
            )
        else:
            logger.info(f"Nothing to be evaluated for {active_logical_class}")
    logger.info("Preparing lambda evaluation metrics...")
    detections_pd = pd.concat(detections_pds, copy=False, ignore_index=True, sort=False)
    out_of_vocab_metrics = pd.concat(
        out_of_vocab_metrics_list, copy=False, ignore_index=True, sort=False
    )
    lambda_metrics_pd = prepare_lambda_metrics(detections_pd, df_apps)
    all_metrics_pd = pd.merge(
        out_of_vocab_metrics, lambda_metrics_pd, how="outer", on="label"
    )
    logger.info(f"Final metrics:\n{all_metrics_pd.to_string()}")

    if etl_pd_new_data.size:
        write_to_lambda_cache(
            etl_pd_new_data,
            customer_id,
            "etlBins",
            new_watermark_ts,
            logger,
        )

    if detections_pd.size:
        detection_insert_sql, rows = prepare_detections_for_export(
            detections_pd, end_time, logger
        )

        if not is_integration_test():
            write_detections_to_rds(detection_insert_sql, rows, customer_id, logger)

        write_to_lambda_cache(
            detections_pd, customer_id, "detections", new_watermark_ts, logger
        )
        return detections_pd, all_metrics_pd

    return None, all_metrics_pd
