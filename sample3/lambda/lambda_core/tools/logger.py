import logging
import time

LOGFORMAT = (
    "[%(levelname)s]\t%(asctime)s.%(msecs)dZ\t%(aws_request_id)s\t%(filename)s:%(funcName)s:%("
    "lineno)d\t%(message)s\n"
)


# Using a custom adaptor to ingest customer_id
class CustomAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        log_statement = "[" + str(self.extra["customer_id"]) + "] " + str(msg)
        return log_statement, kwargs


class AppLogger:
    # Method is added for allowing pre-commit hooks to pass
    @staticmethod
    def info(line: str) -> None:
        pass

    @staticmethod
    def _get_configured_logger(
        customer_id: str, log_level: int
    ) -> logging.LoggerAdapter:
        # Get root logger
        root = logging.getLogger()

        # Log format
        root_handler = root.handlers[0]
        log_format = UTCFormatter(LOGFORMAT, "%Y-%m-%dT%H:%M:%S")
        root_handler.setFormatter(log_format)

        # Set log level
        root.setLevel(log_level)

        # Add a custom adaptor to add customer ID to message
        adaptor = CustomAdapter(logger=root, extra={"customer_id": customer_id})
        return adaptor

    # Using new to return Logger after configuration
    def __new__(cls, customer_id="null", log_level=logging.INFO):
        return cls._get_configured_logger(customer_id, log_level)


class UTCFormatter(logging.Formatter):
    converter = time.gmtime
