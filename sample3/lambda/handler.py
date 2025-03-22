import json
from datetime import datetime

from lambda_core.definitions.request import UserRequest
from lambda_core.module_logic.inputs import prepare_detections_for_export
from lambda_core.module_logic.main import main_logic
from lambda_core.tools.logger import AppLogger as Logger
from lambda_core.tools.utils import is_integration_test
from lambda_core.module_logic.inputs import write_metrics_to_s3


def main(event, context):
    logger = None
    user_request = UserRequest(event)

    try:
        user_request.parse()
    except ValueError as e:
        Logger().error(f"Unable to parse event: {event}")
        raise e

    try:
        if user_request.body:
            data = user_request.body
        else:
            Logger().error("'payload' in main 'body' of event object is empty")
            raise ValueError("'payload' in main 'body' of event object is empty")

        # We initialize logger() with customer_id
        logger = Logger(customer_id=data["customerId"])
        logger.info(f"input_event: {event}")
        detections_pd, lambda_metrics_pd = main_logic(data, logger)

        if lambda_metrics_pd is not None:
            logger.info("Writing metrics to s3...")
            try:
                write_metrics_to_s3(lambda_metrics_pd, data["customerId"])
            except Exception as exp:
                logger.warn("Error writing metrics to s3...")
                logger.exception(exp)

        detections = (
            detections_pd.to_dict(orient="records") if detections_pd is not None else []
        )
        logger.info(f"Detections : {detections}")

    except Exception as exp:
        if logger is None:
            Logger().exception(exp)
        else:
            logger.exception(exp)
        raise exp

    resp = None
    if is_integration_test():
        detection_insert_sql = ""
        rows = []
        if detections_pd is not None:
            end_time = datetime.fromtimestamp(data["endTime"])
            detection_insert_sql, rows = prepare_detections_for_export(
                detections_pd, end_time, logger
            )
        resp = {
            "statusCode": 200,
            "body": json.dumps({"sql": detection_insert_sql, "rows": str(rows)}),
        }
    return resp
