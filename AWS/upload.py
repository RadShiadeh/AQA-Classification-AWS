import logging
import boto3
from botocore.exceptions import ClientError
import os
import dotenv


global s3_client

s3_client = boto3.client('s3')


def upload_file_(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    if object_name is None:
        object_name = os.path.basename(file_name)

    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False, None
    
    return True, response


def upload_from_dir(directory, bucket_name, prefix=""):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            object_name = os.path.join(prefix, filename)
            if not check_obj_exists(bucket_name, object_name):
                success, res = upload_file_(file_path, bucket_name, object_name)
                print(f"the upload of {object_name} was successful? {success}, and the response was: {res}\n")
            else:
                print(f"dups: {object_name} already exists")


def check_obj_exists(bucket_name, obj_file_name):
    try:
        s3_client.head_object(Bucket= bucket_name, Key=obj_file_name)
        return True
    
    except ClientError as e:
        if e.response['Error']['Code'] == "404":
            return False
        else:
            return e



def main():
    dotenv.load_dotenv()

    squats_bucket_name = os.getenv("SQUATS_BUCKET_NAME")
    ohp_bucket_name = os.getenv("OHP_BUCKET_NAME")
    barbell_row_bucket_name = os.getenv("BARBELL_ROW_BUCKET_NAME")

    squats_file_path = "../RawData/squat_unlabled/videos"
    ohp_file_path = "../RawData/OHP_unlabled/videos"
    barbell_row_file_path = "../RawData/barbellRow_labeled/barbellrow_images_raw"

    upload_from_dir(squats_file_path, squats_bucket_name, prefix="squat") #prefix_number_aug
    upload_from_dir(ohp_file_path, ohp_bucket_name, "ohp")
    upload_from_dir(barbell_row_file_path, barbell_row_bucket_name, "barbell_row")


if __name__ == "__main__":
    main()