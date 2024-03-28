import csv
import glob
import json
import logging
import os
import re
import shlex
import string
import subprocess
import tempfile
from datetime import datetime
from functools import cache

import jiwer
import whisper
from pydub import AudioSegment

logging.basicConfig(
    filename="report.log",
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.ERROR,
)

whisper_combinations = [
    {
        "model": "large",
        "beam_size": 5,
        "patience": 1.0,
        "condition_on_previous_text": True,
    },
    {
        "model": "large",
        "beam_size": 5,
        "patience": 1.0,
        "condition_on_previous_text": False,
    },
    {
        "model": "large",
        "beam_size": 5,
        "patience": 2.0,
        "condition_on_previous_text": True,
    },
    {
        "model": "large",
        "beam_size": 5,
        "patience": 2.0,
        "condition_on_previous_text": False,
    },
    {
        "model": "large",
        "beam_size": 10,
        "patience": 1.0,
        "condition_on_previous_text": True,
    },
    {
        "model": "large",
        "beam_size": 10,
        "patience": 1.0,
        "condition_on_previous_text": False,
    },
    {
        "model": "large",
        "beam_size": 10,
        "patience": 2.0,
        "condition_on_previous_text": True,
    },
    {
        "model": "large",
        "beam_size": 10,
        "patience": 2.0,
        "condition_on_previous_text": False,
    },
]

preprocessing_combinations = [
    "afftdn=nr=10:nf=-25:tn=1",
    "afftdn=nr=10:nf=-25:tn=1,volume=4",
    "anlmdn,volume=4",
    "highpass=200,lowpass=3000,afftdn",
    "volume=4",
    "speechnorm",
]


def run():
    try:
        write_headers = True
        files_with_transcript = get_files()
        for file in files_with_transcript:
            for combination in whisper_combinations:
                start_time = datetime.now()
                whisper_fields = run_whisper(file, combination)
                fields = {
                    "file": os.path.basename(file),
                    "duration": get_runtime(start_time),
                }
                fields.update(combination)
                fields.update(whisper_fields)
                with open(
                    f"{datetime.now().date()}-spreadsheet.csv", mode="a", newline=""
                ) as spreadsheet:
                    writer = csv.DictWriter(spreadsheet, fieldnames=fields.keys())
                    if write_headers:
                        writer.writeheader()
                        write_headers = False
                    writer.writerow(fields)
    except Exception as e:
        logging.error(e)


def run_preprocessing():
    try:
        write_headers = True
        files_with_transcript = get_files()
        for file in files_with_transcript:
            for combination in preprocessing_combinations:
                start_time = datetime.now()
                preprocessed_file = (
                    os.path.basename(file).rsplit(".", 1)[0]
                    + "_filter_"
                    + re.sub(r"[^A-Za-z0-9]", "", combination)
                    + ".wav"
                )
                subprocess.Popen(
                    f"ffmpeg -i {file} -af {combination} {preprocessed_file}",
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                ).communicate()
                whisper_fields = run_whisper(
                    preprocessed_file,
                    {"beam_size": 5, "patience": 1, "condition_on_previous_text": True},
                    file,
                )
                fields = {
                    "file": os.path.basename(file),
                    "duration": get_runtime(start_time),
                }
                fields.update({"ffmpeg filter": combination})
                fields.update(whisper_fields)
                os.remove(preprocessed_file)
                csv_filename = f"{datetime.now().date()}-preprocessing-spreadsheet.csv"
                write_headers = write_rows(write_headers, fields, csv_filename)
    except Exception as e:
        logging.error(e)


def write_rows(write_headers, fields, filename):
    with open(filename, mode="a", newline="") as spreadsheet:
        writer = csv.DictWriter(spreadsheet, fieldnames=fields.keys())
        if write_headers:
            writer.writeheader()
            write_headers = False
        writer.writerow(fields)
    return write_headers


def get_files():
    folder = "/home/whisper/Documents/pilot-data/bags/*"
    files_with_transcript = []
    folders = glob.glob(folder + os.path.sep)
    files = [glob.glob("{}data/content/*_sl*.m*".format(folder)) for folder in folders]
    for folder in files:
        for file in folder:
            if (
                len(
                    glob.glob(f"{file.rsplit('.', 1)[0].replace('_sl', '')}*script.txt")
                )
                > 0
            ):
                files_with_transcript.append(file)
                break
    return files_with_transcript


def get_runtime(start_time):
    time_difference = datetime.now() - start_time
    hours, remainder = divmod(time_difference.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}:{int(minutes)}:{int(seconds)}"


def run_whisper(file, options, output_dir="outputs"):
    model_name = options["model"]
    model = load_model(model_name)
    language = get_language(file, model_name)
    audio = load_audio(file)
    result = whisper.transcribe(
        audio=audio,
        model=model,
        word_timestamps=True,
        language=language,
        beam_size=options["beam_size"],
        patience=options["patience"],
        condition_on_previous_text=options["condition_on_previous_text"],
    )
    if os.path.exists(output_dir) is False:
        os.mkdir(output_dir)
    string_options = "_".join([f"{key}={value}" for key, value in options.items()])
    output_filename = f"{os.path.basename(file)}-{string_options}.json"
    with open(os.path.join(output_dir, output_filename), "w") as f:
        f.write(json.dumps(result))
    hypothesis = clean_text(result["text"])
    reference_files = glob.glob(
        f"{file.rsplit('.', 1)[0].replace('_sl', '')}*script.txt"
    )
    find_file = list(filter(lambda x: "_{}".format(language) in x, reference_files))
    reference_file = find_file if len(find_file) > 0 else reference_files
    reference = clean_text(open(reference_file[0]).read())
    output = jiwer.process_words(reference, hypothesis)
    new_dict = {
        "wer": output.wer,
        "mer": output.mer,
        "wil": output.wil,
        "wip": output.wip,
        "hits": output.hits,
        "substitutions": output.substitutions,
        "insertions": output.insertions,
        "deletions": output.deletions,
    }
    new_dict["language"] = language
    return new_dict


def get_language(file, model_name):
    model = load_model(model_name)
    silences = get_silences(file)
    n_mels = 128 if model_name == "large" else 80

    if len(silences) > 0 and int(silences[0]["start_silence"]) == 0:
        audio_segment = AudioSegment.from_file(file)
        start_time = silences[0]["end_silence"] * 1000
        end_time = (30 + silences[0]["end_silence"]) * 1000
        clip = audio_segment[start_time:end_time]
        with tempfile.NamedTemporaryFile() as tmp:
            clip.export(tmp.name, format="wav")
            audio = load_audio(tmp.name)
    else:
        audio = load_audio(file)

    audioclip = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audioclip, n_mels=n_mels).to(model.device)
    _, probs = model.detect_language(mel)

    return max(probs, key=probs.get)


def clean_text(text):
    text = text.replace("\n", " ").replace("  ", " ")
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    return text


def get_silences(file):
    file = shlex.quote(file)
    p = subprocess.Popen(
        "ffmpeg -i {} -af 'volumedetect' -vn -sn -dn -f null /dev/null".format(file),
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).communicate()
    detectvolume = p[-1].decode("utf-8")
    meanvolume = ffmpegcontentparse(detectvolume, "mean_volume")
    volume = meanvolume - 1 if meanvolume > -37 else -37
    p2 = subprocess.Popen(
        "ffmpeg -i {} -af silencedetect=n={}dB:d=0.5 -f null -".format(file, volume),
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).communicate()
    startendsilences = []
    for item in p2[-1].decode("utf-8").split("\n"):
        if "silence_start" in item:
            startendsilences.append(
                {"start_silence": ffmpegcontentparse(item, "silence_start")}
            )
        elif "silence_end" in item:
            startendsilences[-1] = startendsilences[-1] | {
                "end_silence": ffmpegcontentparse(item.split("|")[0], "silence_end"),
                "duration": ffmpegcontentparse(item, "silence_duration"),
            }

    return startendsilences


def ffmpegcontentparse(content, field):
    lines = content.split("\n")
    correctline = [idx for idx, s in enumerate(lines) if field in s][0]
    get_value = lines[correctline].split(":")[-1]
    value_as_float = float(re.sub(r"[^0-9\-.]", "", get_value, 0, re.MULTILINE).strip())
    return value_as_float


@cache
def load_model(model_name):
    # cache the response since it takes some time to load
    return whisper.load_model(model_name)


@cache
def load_audio(file):
    return whisper.load_audio(file)
