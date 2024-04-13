import datetime
import functools
import json
import logging
import os
import pathlib
import time
import uuid

import boto3
import botocore
import dotenv
import requests
import tqdm

from . import utils

dotenv.load_dotenv()


def run(output_dir, manifest):
    results = []
    for file_metadata in tqdm.tqdm(utils.get_data_files(manifest), desc="aws".ljust(10)):
        file_metadata["run_count"] = len(results) + 1
        file = file_metadata["media_filename"]

        if file_metadata["media_language"] != file_metadata["transcript_language"]:
            logging.info("skipping since google doesn't support translation")
            continue

        logging.info("transcribing with aws %s", file)

        start_time = datetime.datetime.now()
        transcription = transcribe(file_metadata)
        runtime = utils.get_runtime(start_time)

        result = utils.compare_transcripts(
            file_metadata, transcription, "aws", output_dir
        )

        result["runtime"] = runtime

        with open(os.path.join(output_dir, f"{result['run_id']}.json"), "w") as fh:
            json.dump(transcription, fh, ensure_ascii=False)

        logging.info("result: %s", result)
        results.append(result)

    csv_filename = os.path.join(output_dir, "report-aws.csv")
    utils.write_report(results, csv_filename)


def transcribe(file_metadata):
    # upload media file to a bucket
    s3_file = upload_file(file_metadata["media_filename"])

    # create the transcription job
    scribe = get_client("transcribe")
    job_name = str(uuid.uuid1())
    scribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={"MediaFileUri": s3_file},
        IdentifyLanguage=True,
    )

    # wait for the job to be complete
    job = wait_for_job(scribe, job_name)

    # fetch the results
    url = job["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]

    return requests.get(url).json()


def upload_file(file):
    path = pathlib.Path(file)
    aws_region = os.environ.get("AWS_REGION")
    bucket_name = os.environ.get("AWS_TRANSCRIBE_S3_BUCKET")

    s3 = get_client("s3")

    try:
        if aws_region:
            s3.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={"LocationConstraint": aws_region},
            )
        else:
            s3.create_bucket(Bucket=bucket_name)
    except botocore.exceptions.ClientError as error:
        # it's ok if the bucket already exists
        if error.response["Error"]["Code"] != "BucketAlreadyOwnedByYou":
            raise error

    s3.upload_file(path, bucket_name, path.name)

    return f"s3://{bucket_name}/{path.name}"


def get_client(service_name):
    return get_session().client(service_name)


@functools.cache
def get_session():
    config = {}

    aws_profile = os.environ.get("AWS_PROFILE")
    if aws_profile:
        config["profile_name"] == aws_profile

    aws_region = os.environ.get("AWS_REGION")
    if aws_region:
        config["region_name"] = aws_region

    return boto3.session.Session(**config)


def wait_for_job(scribe, job_name, wait_seconds=1):
    time.sleep(wait_seconds)
    job = scribe.get_transcription_job(TranscriptionJobName=job_name)
    if job["TranscriptionJob"]["TranscriptionJobStatus"] == "COMPLETED":
        return job
    else:
        return wait_for_job(scribe, job_name, wait_seconds**2)
