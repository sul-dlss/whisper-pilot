import datetime
import os

from . import utils


def run(bags_dir, output_dir):
    results = []
    for file in utils.get_files(bags_dir):
        hypothesis = transcribe(file)
        reference = utils.get_reference_file(file)
        result = utils.compare_transcripts(reference, hypothesis)
        results.append(result)

    csv_filename = os.path.join(
        output_dir, f"{datetime.now().date()}-aws-spreadsheet.csv"
    )
    utils.write_report(results, csv_filename)


def transcribe(media_file):
    return {
        "language": "en_US",
        "transcript": "This is a test for whisper reading in English.",
    }
