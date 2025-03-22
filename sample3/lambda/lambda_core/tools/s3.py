""""Helpers to work with S3 objects efficiently.

The listing functions return objects by default. Use the `_keys` variants to
only get keys.

The base functions do a simple listing. Use the `_parallel` variants to list
multiple prefixes in parallel, which is often a lot faster if you have at least
a few prefixes. For instance, you can split a prefix over 24 sub-prefixes
corresponding to the hour of the day. Splitting into sub-prefixes is not
automated and must be provided as input. See the documentation strings for
examples.

``s3_list_parallel`` uses threads by default, which have no memory or startup
overhead and are a good choice in general. If you have a lot of memory, consider
setting ``pool_context`` to ``spawn`` instead, which can be up to 50% faster
when using large pool sizes. See the documentation of ``s3_list_parallel``.

All functions return iterators, so you can start doing some work while listing
is still ongoing. If you need a list, call `list` of the iterator to materialize
it.
"""
import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import get_context
from typing import Callable, Dict, Iterator, List, Optional, Tuple, TypeVar, cast
from urllib.parse import urlparse

import backoff
import boto3
from botocore.client import BaseClient
from lambda_core.definitions.constants import S3_LIST_RETRY_COUNT

T = TypeVar("T")
TransformFun = Callable[[Dict], Optional[T]]

_DEFAULT_POOL_SIZE = 6
_MAX_POOL_SIZE = 24


@backoff.on_exception(backoff.expo, Exception, max_tries=S3_LIST_RETRY_COUNT)
def get_s3_object(s3_client: BaseClient, key: str, bucket: str):
    return s3_client.get_object(Key=key, Bucket=bucket)


def s3_move_file(s3_client: BaseClient, bucket: str, src_key: str, dst_key: str) -> None:
    s3_client.copy({"Bucket": bucket, "Key": src_key}, bucket, dst_key)
    s3_client.delete_object(Bucket=bucket, Key=src_key)


def s3_exists(
    url: Optional[str] = None,
    bucket: Optional[str] = None,
    prefix: Optional[str] = None,
) -> bool:
    """Check if a prefix exists in s3, i.e. there are objects under it.

    Either an URL (`s3://bucket/prefix`) or the keyword arguments `bucket`,
    `prefix` must be provided.

    Parameters
    ----------
    url
        S3 URL for the bucket and prefix to check. Can be omitted
        if ``bucket``, ``prefix`` are passed in.
    bucket
        S3 bucket name. Can be omitted if ``url`` is passed in.
    prefix
        Object prefix (without leading slash) to check. Can be omitted
        if ``url`` is passed in.

    Returns
    -------
    res
        True if there are objects under the given prefix; false, otherwise.
    """
    assert (bucket and prefix) or url, "Either an URL or a (bucket, prefix) must be provided"
    if url:
        bucket, prefix = _parts_from_url(url)

    bucket, prefix = cast(str, bucket), cast(str, prefix)
    s3 = boto3.client("s3")

    if prefix.startswith("/"):
        prefix = prefix[1:]

    # MaxKeys=1, because we only check for the existence of an object
    # with this prefix. We don't need to load everything.
    objects = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    return objects["KeyCount"] > 0


def s3_list(
    url: Optional[str] = None,
    bucket: Optional[str] = None,
    prefix: Optional[str] = None,
    transform: Optional[TransformFun] = None,
) -> Iterator[T]:
    """List object information from S3.

    For each object, a dictionary is returned with all object properties.
    Use ``s3_list_keys`` to only list keys.

    Parameters
    ----------
    url
        S3 URL for the bucket and prefix to list. Can be omitted
        if ``bucket``, ``prefix`` are passed in.
    bucket
        S3 bucket name. Can be omitted if ``url`` is passed in.
    prefix
        Object prefix (without leading slash) to list. Can be omitted
        if ``url`` is passed in.
    transform
        Optional transformation function to produce a result from the
        object dictionnary - e.g., you can use it to obtain the object
        size or other properties.

        You can use any properties
        [documented in ``boto3``]
        (https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html#S3.Client.list_objects),
        as well as the ``Bucket`` and ``Prefix`` properties.

    Returns
    -------
    By default, return one dictionary per object in the format of boto3.

    If `transform` is specified, objects are transformed using the
    function. When `None` is returned, the object is dropped.

    Examples
    --------
    >>> s3_list("s3://bucket/some/object/prefix")
    [{"Key": "...", ...}]

    >>> s3_list(bucket="bucket", prefix="some/object/prefix")
    [{"Key": "...", ...}]

    >>> s3_list("s3://bucket/prefix", transform=lambda object: object["Key"])
    ["prefix/object1", "..."]

    >>> s3_list(
    >>>    "s3://bucket/prefix",
    >>>    transform=lambda object: object["Key"] if object["Key"].endswidth(".xml") else None)
    >>> )
    []
    """
    assert (bucket and prefix) or url, "Either an URL or a (bucket, prefix) must be provided"
    if url:
        bucket, prefix = _parts_from_url(url)

    session = boto3.Session()
    s3_client = session.client("s3")

    paginator = s3_client.get_paginator("list_objects")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    page_iterator = pages.search("Contents")
    assert page_iterator is not None

    for item in page_iterator:
        if item is None:
            continue

        item["Bucket"] = bucket
        item["Prefix"] = prefix

        if transform is not None:
            item = transform(item)

        if item is not None:
            yield item


def _s3_list_materialized(url: str, transform: Optional[TransformFun] = None) -> List[T]:
    return list(s3_list(url, transform=transform))


def s3_list_keys(
    url: Optional[str] = None,
    bucket: Optional[str] = None,
    prefix: Optional[str] = None,
    full_urls: bool = False,
) -> Iterator[str]:
    """List object keys from S3.

    Unlike ``s3_list``, this function only return object keys.

    Parameters
    ----------
    full_urls
        When set, return full S3 URLs instead of only object keys.

    See Also
    --------
    See ``s3_list``.

    Examples
    --------
    >>> s3_list_keys("s3://bucket/some/object/prefix")
    ["some/object/prefix/key1", ...]

    >>> s3_list_keys("s3://bucket/some/object/prefix", full_urls=True)
    ["s3://bucket/some/object/prefix/key1", ...]
    """
    return s3_list(url, bucket, prefix, _get_object_transform(full_urls))


def s3_list_parallel(
    urls: Optional[List[str]] = None,
    bucket: Optional[str] = None,
    prefixes: Optional[List[str]] = None,
    pool_size: int = _DEFAULT_POOL_SIZE,
    transform: Optional[TransformFun] = None,
    pool_context: str = "asyncio",
) -> Iterator[T]:
    """List object information from S3.

    For each object, a dictionnary is returned with all object properties.
    Use ``s3_list_keys_parallel`` to only list keys.

    Parameters
    ----------
    urls
        S3 URLs for the bucket and prefixes to list. Can be omitted
        if ``bucket``, ``prefixes`` are passed in. URLs will be listed
        in parallel.
    bucket
        S3 bucket name. Can be omitted if ``url`` is passed in.
    prefixes
        Object prefixes (without leading slash) to list. Can be omitted
        if ``urls`` is passed in. Prefixes will be listed in parallel.
    pool_size
        Number of pool workers to use.
    transform
        Optional transformation function to produce a result from the
        object dictionary - e.g., you can use it to obtain the object
        size or other properties.
    pool_context
        A valid [pool context]
        (https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods),
        or the special value ``asyncio``:

        - When a multiprocessing pool context is specified, multiprocessing
          is used.

          multiprocessing has more overhead. Memory usage will increase linearly with
          pool size. However, it can be faster when there are a lot of objects to list
          as separate processes are not subject to the Python global interpreter lock.
          Within multiprocessing:

            - ``fork`` can be marginally faster to start up, but it is not significant
              in practice, and it tends to use more memory.
           - ``spawn`` should be preferred, as it tends to use less memory for similar
              performance.

        - When the context is set to ``asyncio``, ``asyncio`` with a thread pool
          executor is used. ``asyncio`` has almost no memory or startup overhead.

        The default is ``asyncio`` which is more conservative. If you have a lot of
        memory available, consider using ``spawn``.

    See Also
    --------
    See ``s3_list`` for documentation on `transform` and examples.

    Examples
    --------
    >>> s3_list_parallel(["s3://bucket/data/1", "s3://bucket/data/2"])
    [{"Key": "...", ...}]

    >>> s3_list_parallel(bucket="bucket", prefixes=["data/1", "data/2"])
    [{"Key": "...", ...}]

    >>> # Split a base prefix into 24 sub-prefixes by hour
    >>> prefixes = [os.path.join("my/prefix/2020/10/01", f"{i:02d}/") for i in range(24)]
    >>> s3_list_parallel(bucket="bucket", prefixes=prefixes)
    """
    assert (bucket and prefixes) or urls, "Either URLs or a (bucket, prefixes) must be provided"
    assert pool_size <= _MAX_POOL_SIZE, "Pool size must be reasonable"

    if not urls:
        bucket = cast(str, bucket)
        prefixes = cast(List[str], prefixes)

        urls = [f"s3://{bucket}/{prefix}" for prefix in prefixes]

    if pool_context == "asyncio":
        with ThreadPoolExecutor(max_workers=pool_size) as pool_executor:
            event_loop = asyncio.get_event_loop()

            async def async_list(executor):
                loop = asyncio.get_event_loop()
                tasks = []

                for url in urls:
                    tasks.append(
                        loop.run_in_executor(
                            executor,
                            functools.partial(_s3_list_materialized, url=url, transform=transform),
                        )
                    )

                completed, _ = await asyncio.wait(tasks)
                return [t.result() for t in completed]

            iterators = event_loop.run_until_complete(async_list(pool_executor))

    else:
        with get_context(pool_context).Pool(pool_size) as pool:
            # We cannot consume from a generator, so we need to materialize the
            # results from each chunk.
            iterators = pool.map(
                functools.partial(_s3_list_materialized, transform=transform),
                urls,
                chunksize=1,
            )

    return (item for iterator in iterators for item in iterator)


@backoff.on_exception(backoff.expo, Exception, max_tries=S3_LIST_RETRY_COUNT)
def s3_list_keys_parallel(
    urls: Optional[List[str]] = None,
    bucket: Optional[str] = None,
    prefixes: Optional[List[str]] = None,
    pool_size: int = _DEFAULT_POOL_SIZE,
    pool_context: str = "asyncio",
    full_urls: bool = False,
) -> Iterator[str]:
    """List object keys from S3, in parallel.

    Unlike ``s3_list``, this function only return object keys.

    Parameters
    ----------
    full_urls
        When set, return full S3 URLs instead of only object keys.

    See Also
    --------
    `s3_list_parallel`, `s3_list_keys`.
    """
    return s3_list_parallel(
        urls,
        bucket,
        prefixes,
        pool_size,
        _get_object_transform(full_urls),
        pool_context,
    )


def _get_object_transform(full_urls: bool) -> Callable[[Dict], str]:
    if full_urls:
        return _object_url
    else:
        return _object_key


def _object_url(object: Dict) -> str:
    return f"s3://{object['Bucket']}/{object['Key']}"


def _object_key(object: Dict) -> str:
    return object["Key"]


def _parts_from_url(url: str) -> Tuple[str, str]:
    url_parsed = urlparse(url)

    return url_parsed.netloc, url_parsed.path[1:]
