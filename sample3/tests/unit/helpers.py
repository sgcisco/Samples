import boto3


MOCK_BUCKET_NAME = "rawdata"
MOCK_S3_REGION = "us-east-1"

FILE_S3_FOLDER = "testuser/data/"


def s3_create_bucket():
    conn = boto3.resource("s3", region_name=MOCK_S3_REGION)
    conn.create_bucket(Bucket=MOCK_BUCKET_NAME)


def s3_get_client():
    return boto3.client("s3", region_name=MOCK_S3_REGION)


def s3_upload_file_object(s3_client, src_file_name, dst_file_name):
    with open(src_file_name, "rb") as f:
        s3_client.upload_fileobj(f, Bucket=MOCK_BUCKET_NAME, Key=dst_file_name)


def s3_upload_file(s3_client, file_name, data):
    s3_client.put_object(Bucket=MOCK_BUCKET_NAME, Key=file_name, Body=data)


def s3_read_file(file_name):
    conn = boto3.resource("s3", region_name=MOCK_S3_REGION)
    return conn.Object(MOCK_BUCKET_NAME, file_name).get()["Body"].read().decode("utf-8")
