import datetime
import json
import logging
import os
import subprocess
import tempfile

import tqdm
from google.api_core.exceptions import NotFound
from google.cloud import speech, storage
from google.protobuf.json_format import MessageToDict

from . import utils


def run(output_dir):
    results = []
    for file_metadata in tqdm.tqdm(utils.get_data_files(), desc="google".ljust(10)):
        file_metadata["run_count"] = len(results) + 1
        file = file_metadata["media_filename"]

        if file_metadata["media_language"] != file_metadata["transcript_language"]:
            logging.info("skipping since google doesn't support translation")
            continue

        logging.info(f"running google speech-to-text with {file}")

        start_time = datetime.datetime.now()
        transcription = transcribe(file_metadata)
        runtime = utils.get_runtime(start_time)

        result = utils.compare_transcripts(
            file_metadata, transcription, "google", output_dir
        )
        result["runtime"] = runtime

        with open(os.path.join(output_dir, f"{result['run_id']}.json"), "w") as fh:
            json.dump(transcription, fh, ensure_ascii=False)

        logging.info(f"result: {result}")
        results.append(result)

    csv_filename = os.path.join(output_dir, "report-google.csv")
    utils.write_report(results, csv_filename)


def transcribe(file_metadata):
    """
    Sends the media file using Google Speech API, and returns the result as a dict.
    """

    # convert the media file to single channel wav and upload to google cloud
    wav_file = convert_to_wav(file_metadata["media_filename"])
    blob_uri = copy_file(wav_file)
    audio = speech.RecognitionAudio(uri=blob_uri)

    logging.info(f"starting speech-to-text job for {wav_file}")

    # unlike aws and whisper, google v1 speech API needs to know the language
    # v2 appears to be different but I couldn't get it to work properly
    # automatic language detection will be something we want to explore if we
    # decide to use Google

    language = file_metadata["media_language"]
    config = speech.RecognitionConfig(language_code=language)

    # send the transcription job to google
    client = speech.SpeechClient()
    operation = client.long_running_recognize(audio=audio, config=config)
    response = operation.result(timeout=60 * 60 * 2)

    # remove the temporary wav file
    os.remove(wav_file)

    return MessageToDict(response._pb)


def copy_file(media_file):
    bucket_name = os.environ.get("GOOGLE_TRANSCRIBE_GCS_BUCKET")
    logging.info(f"copying {media_file} to google storage bucket {bucket_name}")
    storage_client = storage.Client()

    try:
        bucket = storage_client.get_bucket(bucket_name)
    except NotFound:
        bucket = storage_client.create_bucket(bucket_name)

    filename = os.path.basename(media_file)
    blob = bucket.blob(filename)
    blob.upload_from_filename(media_file)

    return f"gs://{bucket_name}/{filename}"


def convert_to_wav(media_file):
    temp_dir = tempfile.gettempdir()
    wav_file = os.path.join(temp_dir, os.path.basename(media_file))
    wav_file, ext = os.path.splitext(wav_file)
    wav_file = f"{wav_file}.wav"

    logging.info(f"ffmpeg converting {media_file} to {wav_file}")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "panic", "-i", media_file, "-ac", "1", wav_file]
    )
    return wav_file
