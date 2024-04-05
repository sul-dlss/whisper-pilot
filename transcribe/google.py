import datetime
import json
import logging
import os
import subprocess
import tempfile
from collections import Counter

import tqdm
from google.api_core.exceptions import NotFound
from google.cloud import speech, storage

from . import utils


def run(bags_dir, output_dir):
    results = []
    for file in tqdm.tqdm(utils.get_files(bags_dir)):
        logging.info(f"running google speech-to-text with {file}")

        start_time = datetime.datetime.now()
        transcription = transcribe(file)
        runtime = utils.get_runtime(start_time)

        with open(
            os.path.join(output_dir, f"{os.path.basename(file)}-google.json"), "w"
        ) as fh:
            json.dump(transcription, fh, ensure_ascii=False)

        reference = utils.get_reference_file(file, "en")
        result = utils.compare_transcripts(reference, transcription["text"])
        result["language"] = transcription["language"]
        result["file"] = os.path.basename(file)
        result["runtime"] = runtime
        logging.info(f"result: {result}")
        results.append(result)

    csv_filename = os.path.join(output_dir, "report-google.csv")
    utils.write_report(results, csv_filename)


def transcribe(media_file):

    # convert the media file to single channel wav and upload to google cloud
    wav_file = convert_to_wav(media_file)
    blob_uri = copy_file(wav_file)
    audio = speech.RecognitionAudio(uri=blob_uri)

    # send the transcription job to google
    logging.info(f"starting speech-to-text job for {wav_file}")
    config = speech.RecognitionConfig(language_code="en-US", model="latest_long")
    client = speech.SpeechClient()
    operation = client.long_running_recognize(config=config, audio=audio)
    response = operation.result(timeout=60 * 60 * 2)

    # join together all the text chunks in results
    text = "".join([result.alternatives[0].transcript for result in response.results])

    # get the most common language
    language = Counter(
        [result.language_code for result in response.results]
    ).most_common(1)[0][0]

    # remove the temporary wav file
    os.remove(wav_file)

    return {
        "language": language,
        "text": text,
    }


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
