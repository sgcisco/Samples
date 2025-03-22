import asyncio
import functools
import io
import itertools
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from logging import Logger
from typing import Any, Dict, Iterator, List, Optional, Tuple, ValuesView
from uuid import uuid4

import boto3
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras as extras
from ai_cloud.tools.aws_rds import get_customer_rds_conn
from fastavro import reader
from lambda_core.definitions.constants import (
    CACHE_CLEANUP_TRESHOLD,
    CUSTOMERS_DATASET_BUCKET,
    ENDPOINT_CACHE_CLEANUP_TRESHOLD,
    ENDPOINT_DEVICE_CLASSIFICATION_COLUMNS,
    ENDPOINT_KEY_COLUMNS,
    INFERENCE_INTERVAL,
    INFERENCE_WINDOW,
    LAMBDA_CACHE_BUCKET,
    LAMBDA_CACHE_PREFIX,
    LAMBDA_HISTORY_PREFIX,
    LAMBDA_METRICS_LOG_PATH_KEY,
    RAW_DATA_BUCKET,
    RDS_PRODUCT,
    RDS_ROLE,
    RDS_TABLE_NAME,
    WATERMARK_TS_FORMAT,
)

from lambda_core.definitions.port_scans.port_scans import (
    prepare_app_wise_threshold_data_dict,
)
from lambda_core.module_logic.ml.clf_dataset import ClassificationDataset
from lambda_core.tools.s3 import (
    s3_list,
    s3_list_keys_parallel,
    s3_move_file,
)
from lambda_core.tools.utils import is_integration_test, minutes_range

_DEFAULT_POOL_SIZE = 6


(
    APP_MIN_FLOW_BYTES,
    DEFAULT_MIN_FLOW_CLIENT_BYTES,
    DEFAULT_MIN_FLOW_SERVER_BYTES,
) = prepare_app_wise_threshold_data_dict()


def _dedup_endpoint(rst_rows: List[Any]) -> ValuesView[Tuple[Any, ...]]:
    # This function keeps the first unique ENDPOINT entry
    # defined by a tuple of values at ENDPOINT_KEY_COLUMNS

    d: Dict[Any, Any] = {}
    for row in rst_rows:
        key = tuple([row[c] for c in ENDPOINT_KEY_COLUMNS])

        if (key in d and row["timestamp"] < d[key]["timestamp"]) or key not in d:
            for c in ENDPOINT_DEVICE_CLASSIFICATION_COLUMNS:
                row[c] = list(row[c])
            d[key] = row

    return d.values()


def read_raw_netflow_endpoint(
    customer_id: str,
    start_time: datetime,
    end_time: datetime,
    cur_watermark_ts: datetime,
    logger: Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame, datetime]:
    bucket = RAW_DATA_BUCKET
    prefixes = [
        "{customer_id}/data/{time_prefix}".format(
            customer_id=customer_id, time_prefix=d.strftime("%Y-%m-%dT%H-%M")
        )
        for d in minutes_range(start_time, end_time)
    ]

    try:
        begin = time.perf_counter()

        all_keys = s3_list_keys_parallel(
            bucket=bucket,
            prefixes=prefixes,
        )

    except Exception as exc:
        logger.warn(f"Encountered error at S3 listing '{exc}'")
        raise

    logger.info(f"Prefix listing took {time.perf_counter() - begin:0.4f} seconds")

    out_entities: Dict[str, List[Tuple[str, str]]] = {
        "endpointclassification": [],
        "netflowpacket": [],
    }

    begin = time.perf_counter()

    watermark_ts = start_time
    for key in all_keys:
        if key.endswith(".avro"):
            entity_tokens = key.split("/")
            # We assume the entity is 2 folders away from the "data" folder
            data_index = entity_tokens.index("data")
            entity = entity_tokens[data_index + 2]
            ts = datetime.strptime(entity_tokens[data_index + 1], "%Y-%m-%dT%H-%M-%S.%fZ")
            if entity in out_entities and ts > cur_watermark_ts:
                watermark_ts = max([watermark_ts, ts])
                out_entities[entity].append((bucket, key))
    logger.info(f"Prefixes loading took {time.perf_counter() - begin:0.4f} seconds")
    logger.info(f"Bucket, key pairs per prefix: {out_entities}")

    try:
        endpoint_pd = pd.DataFrame(
            _dedup_endpoint(
                [
                    row
                    for chunk in read_avro_parallel(
                        out_entities["endpointclassification"],
                        "classificationData",
                        logger,
                    )
                    for row in chunk
                ]
            ),
            copy=False,
        )
    except Exception:
        endpoint_pd = pd.DataFrame()

    try:
        nf_pd = pd.DataFrame(
            [
                row
                for chunk in read_avro_parallel(out_entities["netflowpacket"], "records", logger)
                for row in chunk
            ],
            copy=False,
        )
    except Exception:
        nf_pd = pd.DataFrame()

    return (
        nf_pd,
        endpoint_pd,
        watermark_ts,
    )


def _extract_items(d: Dict, col: str) -> Optional[Dict]:
    # This function unpacks the target column in ENDPOINT and NF
    # In case of NF records, it always has one item only
    #   // nf-stream is currently one record only per msg.
    # 	// But, keep array in the interface, in case it is changed later.
    # 	var records []*netflow_avro.NetflowRecord
    # 	records = append(records, record)
    # 	avro.Records = records
    d.update(d[col][0]) if col == "records" else d.update(d[col])
    # Removing non relevant `matchedRules` from `endpointclassification`
    if col == "classificationData":
        d.pop("matchedRules", None)
        d.pop("logicalProfile", None)
        d.pop("classificationTimestamp", None)
        d.pop("macRandomData", None)

        for c in ENDPOINT_DEVICE_CLASSIFICATION_COLUMNS:
            d[c] = tuple(d[c] if d[c] else [])

    d.pop(col, None)
    # Filter some activity records
    if col == "records":
        app = d["appName"]
        if app == "Unknown":
            app = "unknown"

        if (
            app in APP_MIN_FLOW_BYTES["appWiseThresholds"]
            and d["clientBytesSent"]
            >= APP_MIN_FLOW_BYTES["appWiseThresholds"][app]["minFlowClientBytes"]
            and d["serverBytesSent"]
            >= APP_MIN_FLOW_BYTES["appWiseThresholds"][app]["minFlowServerBytes"]
        ) or (
            app not in APP_MIN_FLOW_BYTES["appWiseThresholds"]
            and (
                (
                    d["clientBytesSent"] >= DEFAULT_MIN_FLOW_CLIENT_BYTES
                    and d["serverBytesSent"] >= DEFAULT_MIN_FLOW_SERVER_BYTES
                )
                or (
                    d["clientBytesSent"] >= DEFAULT_MIN_FLOW_SERVER_BYTES
                    and d["serverBytesSent"] >= DEFAULT_MIN_FLOW_CLIENT_BYTES
                )
            )
        ):
            return d
        else:
            return None
    return d


def read_avro_file(
    tuple_list: Tuple[str, str], json_col: str, logger: Logger
) -> List[Dict[Any, Any]]:
    begin = time.perf_counter()
    session = boto3.Session()
    s3_client = session.client("s3")
    tot_bytes = 0
    bucket, prefix = tuple_list
    begin = time.perf_counter()
    response = s3_client.get_object(Key=prefix, Bucket=bucket)
    avro_file = response["Body"].read()
    file_length = len(avro_file)
    logger.info(
        f"Reading avro file {prefix} of size {file_length} took {time.perf_counter() - begin:0.4f} seconds"  # noqa
    )
    tot_bytes += file_length
    begin = time.perf_counter()

    # Despite claiming to accept a IO object the constructor accesses a file specific member
    lines = []
    for line in reader(io.BytesIO(avro_file)):
        rst_line = _extract_items(line, json_col)
        if rst_line is not None:
            lines.append(rst_line)

    logger.info(f"Decoding avro file {prefix} took {time.perf_counter() - begin:0.4f} seconds")

    logger.info(f"{tuple_list} ({tot_bytes} bytes) took {time.perf_counter() - begin:0.4f} seconds")
    return lines


def read_avro_parallel(
    tuple_lists: List[Tuple[str, str]], json_col: str, logger: Logger
) -> List[Any]:
    if tuple_lists:
        with ThreadPoolExecutor(max_workers=_DEFAULT_POOL_SIZE) as pool_executor:
            event_loop = asyncio.get_event_loop()

            async def async_list(executor):
                loop = asyncio.get_event_loop()
                tasks = []

                for tuple_list in tuple_lists:
                    tasks.append(
                        loop.run_in_executor(
                            executor,
                            functools.partial(
                                read_avro_file,
                                tuple_list=tuple_list,
                                json_col=json_col,
                                logger=logger,
                            ),
                        )
                    )

                completed, _ = await asyncio.wait(tasks)
                return [t.result() for t in completed]

            iterators = event_loop.run_until_complete(async_list(pool_executor))

        return [iterator for iterator in iterators]
    return [[]]


def read_s3_in_prefix(
    bucket: str,
    prefix: str,
    entity: str,
    keep_only_last: bool,
    logger: Logger,
) -> Optional[pd.DataFrame]:
    all_keys: list = list(
        s3_list(bucket=bucket, prefix=prefix, transform=lambda object: object["Key"])
    )

    # Only last endpoint cache file is read
    if keep_only_last and all_keys:
        all_keys = [sorted(all_keys, reverse=True)[0]]

    try:
        session = boto3.Session()
        s3_client = session.client("s3")
        out_dfs = []
        for all_key in all_keys:
            begin = time.perf_counter()
            response = s3_client.get_object(Key=all_key, Bucket=bucket)
            parquet_file = response["Body"].read()
            parquet_bytes = io.BytesIO(parquet_file)
            out_dfs.append(pd.read_parquet(parquet_bytes))
            logger.info(
                f"Lambda read for entity {entity} from {LAMBDA_CACHE_BUCKET}/{all_key} took {time.perf_counter() - begin:0.4f} seconds"  # noqa
            )

    except Exception as exc:
        logger.warn(f"Encountered error at S3 retrieve object '{exc}'")
        raise

    try:
        return pd.concat(out_dfs, copy=False, ignore_index=True, sort=False)
    except Exception:
        logger.warn(f"Nothing to read under {bucket}/{prefix}")  # noqa
        return None


def read_lambda_cache(
    customer_id: str, entity: str, logger: Logger, keep_only_last: bool = False
) -> pd.DataFrame:
    lambda_cache_prefix = LAMBDA_CACHE_PREFIX.format(entity=entity, customer_id=customer_id)
    ret = read_s3_in_prefix(
        LAMBDA_CACHE_BUCKET,
        lambda_cache_prefix,
        entity,
        keep_only_last,
        logger,
    )

    return ret


def write_to_lambda_cache(
    data: pd.DataFrame,
    customer_id: str,
    entity: str,
    reference_ts: datetime,
    logger: Logger,
) -> None:
    tmp_file = tempfile.NamedTemporaryFile()
    data.to_parquet(tmp_file.name)
    parquet_bytes = open(tmp_file.name, "rb").read()

    session = boto3.Session()
    s3_client = session.client("s3")

    lambda_cache_prefix = (
        LAMBDA_CACHE_PREFIX.format(entity=entity, customer_id=customer_id)
        + reference_ts.strftime(WATERMARK_TS_FORMAT)
        + ".parquet"
    )

    begin = time.perf_counter()
    s3_client.put_object(Body=parquet_bytes, Bucket=LAMBDA_CACHE_BUCKET, Key=lambda_cache_prefix)
    logger.info(
        f"Lambda write entity {entity} to {LAMBDA_CACHE_BUCKET}/{lambda_cache_prefix} took {time.perf_counter() - begin:0.4f} seconds"  # noqa
    )



def clean_lambda_cache(
    customer_id: str,
    entity: str,
    limit_ts: datetime,
) -> None:
    bucket = LAMBDA_CACHE_BUCKET
    input_prefix = LAMBDA_CACHE_PREFIX.format(entity=entity, customer_id=customer_id)
    output_prefix = LAMBDA_HISTORY_PREFIX.format(entity=entity, customer_id=customer_id)

    all_keys: Iterator = s3_list(
        bucket=bucket, prefix=input_prefix, transform=lambda object: object["Key"]
    )

    session = boto3.Session()
    s3_client = session.client("s3")
    for key in all_keys:
        filename = key.split("/")[-1]
        file_ts = datetime.strptime(filename.rsplit(".", 1)[0], WATERMARK_TS_FORMAT)
        if file_ts < limit_ts:
            dst_key = output_prefix + filename
            s3_move_file(s3_client, bucket, key, dst_key)


def clean_lambda_caches(
    customer_id: str,
    new_watermark_ts: datetime,
    logger: Logger,
) -> None:
    default_cache_cleanup_threshold = new_watermark_ts - CACHE_CLEANUP_TRESHOLD
    endpoint_cache_cleanup_threshold = new_watermark_ts - ENDPOINT_CACHE_CLEANUP_TRESHOLD

    entity_cleanup_threshold = {
        "watermarkTs": default_cache_cleanup_threshold,
        "etlBins": default_cache_cleanup_threshold,
        "endpoint": endpoint_cache_cleanup_threshold,
    }

    for entity, limit_ts in entity_cleanup_threshold.items():
        logger.info(f"Cleaning cache for {entity} while limit timestamp {limit_ts}")
        clean_lambda_cache(customer_id, entity, limit_ts)


def read_watermark(
    customer_id: str,
    end_time: datetime,
    logger: Logger,
) -> datetime:
    # TODO: Check if we should move logic to RDS
    watermark_pd = read_lambda_cache(customer_id, "watermarkTs", logger)
    if watermark_pd is None:
        return end_time - INFERENCE_INTERVAL
    return watermark_pd["timestamp"].max()


def write_watermark(
    watermark_ts: datetime,
    customer_id: str,
    logger: Logger,
) -> None:
    watermark_ts_pd = pd.DataFrame([{"timestamp": watermark_ts}])
    logger.info(f"Write watermark {watermark_ts} for {customer_id} to cache")
    write_to_lambda_cache(watermark_ts_pd, customer_id, "watermarkTs", watermark_ts, logger)


def read_inputs(
    customer_id: str,
    start_time: datetime,
    end_time: datetime,
    cur_watermark_ts: datetime,
    logger: Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, datetime]:
    logger.info(f"Reading inputs between {start_time} and {end_time}")
    begin = time.perf_counter()
    raw_nf_pd, raw_endpoint_pd, watermark_ts = read_raw_netflow_endpoint(
        customer_id, start_time, end_time, cur_watermark_ts, logger
    )
    logger.info(f"Endpoint and NetfLow loading took {time.perf_counter() - begin:0.4f} seconds")

    endpoint_state_pd = read_lambda_cache(customer_id, "endpoint", logger, True)
    if endpoint_state_pd is not None and endpoint_state_pd.size:
        logger.info(f"Endpoint cache data: {len(endpoint_state_pd)} records")

    etl_pd = read_lambda_cache(customer_id, "etlBins", logger)
    if etl_pd is not None and len(etl_pd) > 0:
        etl_pd = etl_pd[etl_pd["timeBin"] > (end_time - INFERENCE_WINDOW)]
        if etl_pd.empty:
            etl_pd = None

    return (
        normalize_nf_data(raw_nf_pd, logger),
        normalize_endpoint_data(raw_endpoint_pd, logger, endpoint_state_pd),
        etl_pd,
        watermark_ts,
    )
