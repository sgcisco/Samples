import os
from datetime import timedelta


def minutes_range(start_time, end_time):
    for n in range(int((end_time - start_time).total_seconds() / 60)):
        yield start_time + timedelta(minutes=n)


def is_integration_test() -> bool:
    """Check if we are running a integration test."""
    return os.environ.get("TEST_UUID", "").startswith("integration") or os.environ.get(
        "TEST_UUID", ""
    ).startswith("debug")
