import boto3
from moto import mock_s3
from unittest.mock import patch

from lambda_core.tools.s3 import (
    s3_exists,
    s3_list,
    s3_list_parallel,
    s3_list_keys,
    s3_list_keys_parallel,
)


@mock_s3
def test_s3_exists():
    bucket = "bucket"
    prefix_with_data = "hello/world/I/have/3/objects"
    prefix_without_data = "no/data/here"

    conn = boto3.resource("s3", region_name="us-east-1")
    conn.create_bucket(Bucket=bucket)

    s3_client = boto3.client("s3")

    s3_client.put_object(Bucket=bucket, Key=f"{prefix_with_data}/obj1", Body="blabla")
    s3_client.put_object(Bucket=bucket, Key=f"{prefix_with_data}/obj2", Body="blabla")
    s3_client.put_object(Bucket=bucket, Key=f"{prefix_with_data}/obj3", Body="blabla")

    assert s3_exists(bucket=bucket, prefix=prefix_with_data)
    assert s3_exists(f"s3://{bucket}/{prefix_with_data}")

    assert not s3_exists(bucket=bucket, prefix=prefix_without_data)
    assert not s3_exists(f"s3://{bucket}/{prefix_without_data}")


# Cannot mock S3 in the parallel case, so we mock the listing function instead.
def _mock_list(url, transform=None):
    objects = ["prefix/03/file.xml.gz", "prefix/03/file.ppt", "prefix/02/file.ppt"]
    return [
        transform({"Key": object}) if transform is not None else {"Key": object}
        for object in objects
        if object.startswith(url.replace("s3://data-bucket/", ""))
    ]


# `moto3` doesn't always mock `boto3.Session` if something else than the
# default session is used. Force the use of the default session everywhere,
# as this one is always mocked by `moto3`.
@mock_s3
@patch("boto3.session")
def test_s3_listing(mock_session, mocker):
    bucket = "data-bucket"
    conn = boto3.resource("s3")
    conn.create_bucket(Bucket=bucket)

    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket, Key="prefix/03/file.xml.gz", Body="dummy")
    s3.put_object(Bucket=bucket, Key="prefix/03/file.ppt", Body="dummy")
    s3.put_object(Bucket=bucket, Key="prefix/02/file.ppt", Body="dummy")

    # Mock, and use `fork` pool context. Although it is less reliable, it is
    # slightly faster for unit testing.
    mocker.patch("lambda_core.tools.s3._s3_list_materialized", _mock_list)

    objs = set(obj["Key"] for obj in s3_list("s3://data-bucket/prefix"))
    objs_prefix = set(
        obj["Key"] for obj in s3_list(bucket="data-bucket", prefix="prefix")
    )
    objs2 = set(
        obj["Key"]
        for obj in s3_list_parallel(["s3://data-bucket/prefix"], pool_context="fork")
    )
    objs3_prefix = set(
        obj["Key"]
        for obj in s3_list_parallel(
            bucket="data-bucket",
            prefixes=["prefix/03", "prefix/02"],
            pool_context="fork",
        )
    )

    assert len(objs) == 3
    assert objs2 == objs3_prefix == objs_prefix

    for pool_context in {"spawn", "asyncio"}:
        assert objs2 == set(
            obj["Key"]
            for obj in s3_list_parallel(
                ["s3://data-bucket/prefix"], pool_context=pool_context
            )
        )

    objs = set(s3_list_keys("s3://data-bucket/prefix"))
    objs2 = set(s3_list_keys("s3://data-bucket/prefix", full_urls=True))
    objs3 = set(
        s3_list_keys_parallel(
            ["s3://data-bucket/prefix/03", "s3://data-bucket/prefix/02"],
            pool_context="fork",
        )
    )

    assert all([url.startswith("s3://data-bucket/") for url in objs2])
    assert len(objs2) == len(objs)

    assert (
        objs
        == objs3
        == {"prefix/03/file.xml.gz", "prefix/03/file.ppt", "prefix/02/file.ppt"}
    )

    assert list(s3_list_keys("s3://data-bucket/xxx")) == []
    assert list(s3_list("s3://data-bucket/", transform=lambda object: None)) == []
