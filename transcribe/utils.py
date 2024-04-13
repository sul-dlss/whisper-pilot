import csv
import datetime
import difflib
import os
import re
import string
import textwrap
from collections import Counter
from pathlib import Path

import jiwer

base_csv_columns = [
    "run_id",
    "druid",
    "file",
    "language",
    "runtime",
    "wer",
    "mer",
    "wil",
    "wip",
    "hits",
    "substitutions",
    "insertions",
    "deletions",
    "diff",
]


def get_data_files():
    rows = []
    data_csv = Path(__file__).parent.parent / "data.csv"
    for row in csv.DictReader(open(data_csv)):
        rows.append(row)
    return rows


def get_runtime(start_time):
    elapsed = datetime.datetime.now() - start_time
    return elapsed.total_seconds()


def write_report(rows, csv_path, extra_cols=[]):
    fieldnames = base_csv_columns.copy()
    if len(extra_cols) > 0:
        fieldnames.extend(extra_cols)
    with open(csv_path, "w") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def compare_transcripts(file, transcript, transcript_type, output_dir):
    """
    Compare the given file (a dictionary of file metadata, a row from data.csv).
    To the given transcript result, and the transcript_type in order to
    differentiate the different ways that results are represented. The
    output_dir is supplied because in addition to returning the comparison an
    HTML diff will be written to the output_dir.
    """
    run_id = f"{file['druid']}-{transcript_type}-{file['run_count']:03}"

    if transcript_type == "google":
        hypothesis, lang = parse_google(transcript)
    elif transcript_type == "aws":
        hypothesis, lang = parse_aws(transcript)
    elif transcript_type == "whisper":
        hypothesis, lang = parse_whisper(transcript)
    else:
        raise Exception("Unknown transcript type: {transcript_type}")

    reference = open(file["transcript_filename"]).readlines()

    stats = jiwer.process_words(clean_text(reference), clean_text(hypothesis))

    diff_file = f"{run_id}.html"
    diff_url = f"https://sul-dlss.github.io/whisper-pilot/{os.path.basename(output_dir)}/{diff_file}"
    diff_path = os.path.join(output_dir, diff_file)
    write_diff(reference, hypothesis, diff_path)

    return {
        "run_id": run_id,
        "druid": file["druid"],
        "file": os.path.basename(file["media_filename"]),
        "wer": stats.wer,
        "mer": stats.mer,
        "wil": stats.wil,
        "wip": stats.wip,
        "hits": stats.hits,
        "substitutions": stats.substitutions,
        "insertions": stats.insertions,
        "deletions": stats.deletions,
        "language": lang,
        "diff": diff_url,
    }


def wrap_lines(lines):
    new_lines = []
    for line in lines:
        new_lines.extend(textwrap.wrap(line.strip(), width=80))
    return new_lines


def write_diff(reference, hypothesis, diff_path):
    from_lines = wrap_lines(reference)
    to_lines = wrap_lines(hypothesis)
    diff = difflib.HtmlDiff().make_file(from_lines, to_lines, "reference", "transcript")
    open(diff_path, "w").writelines(diff)


def parse_google(data):
    lines = [result["alternatives"][0]["transcript"] for result in data["results"]]
    lang_counts = Counter([result["languageCode"] for result in data["results"]])
    lang = lang_counts.most_common(1)[0][0]

    return lines, lang


def parse_whisper(data):
    lines = [segment["text"] for segment in data["segments"]]
    lang = data["language"]

    return lines, lang


def parse_aws(data):
    lines = [t["transcript"] for t in data["results"]["transcripts"]]
    lang = data["results"]["language_code"]

    return lines, lang


def clean_text(lines):
    text = " ".join(lines)
    text = text.replace("\n", " ")
    text = re.sub(r"  +", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    text = text.strip()
    return text
