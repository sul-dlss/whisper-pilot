import json
import logging
import os
import re
import shlex
import subprocess
import tempfile
from datetime import datetime
from functools import cache
from itertools import product

import torch
import tqdm
import whisper
from pydub import AudioSegment

from . import utils

# These are whisper options that we want to perturb. 
#
# The defaults are:
#   beam_size: 5
#   patience: 1
#   condition_on_previous_text: True
#   best_of = 5

whisper_options = {
    "model_name": ["large"],
    "beam_size": [5, 10],
    "patience": [1.0, 2.0],
    "condition_on_previous_text": [True, False],
    "best_of": [5, 10],
}

preprocessing_combinations = [
    "afftdn=nr=10:nf=-25:tn=1",
    "afftdn=nr=10:nf=-25:tn=1,volume=4",
    "anlmdn,volume=4",
    "highpass=200,lowpass=3000,afftdn",
    "volume=4",
    "speechnorm",
]


def run(bags_dir, output_dir):
    results = []
    combinations = list(whisper_option_combinations())
    for file in tqdm.tqdm(utils.get_files(bags_dir), desc="whisper"):
        for options in tqdm.tqdm(combinations, desc=" options", leave=False):
            result = run_whisper(file, options, output_dir)
            logging.info("result: %s", result)
            results.append(result)

    csv_filename = os.path.join(output_dir, "report-whisper.csv")
    utils.write_report(results, csv_filename, extra_cols=["options"])


def run_preprocessing(bags_dir, output_dir):
    results = []
    for file in tqdm.tqdm(utils.get_files(bags_dir), desc="preprocessing"):
        for combination in tqdm.tqdm(
            preprocessing_combinations, desc=" options", leave=False
        ):
            logging.info("preprocessing for file %s: %s", file, combination)
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
            result = run_whisper(
                preprocessed_file,
                {
                    "model_name": "large",
                    "beam_size": 5,
                    "patience": 1,
                    "condition_on_previous_text": True,
                },
                output_dir,
            )
            result["ffmpeg filer"] = combination
            logging.info("result %s", result)
            results.append(result)

            os.remove(preprocessed_file)

    csv_filename = os.path.join(output_dir, "report-whisper-preprocessing.csv")
    utils.write_report(results, csv_filename, extra_cols=["ffmpeg filer"])


def run_whisper(file, options, output_dir="outputs"):
    start_time = datetime.now()
    logging.info("running whisper on %s with options %s", file, options)
    transcription = transcribe(file, options)
    runtime = utils.get_runtime(start_time)

    string_options = "_".join([f"{key}={value}" for key, value in options.items()])
    output_filename = f"{os.path.basename(file)}-{string_options}.json"
    with open(os.path.join(output_dir, output_filename), "w") as f:
        json.dump(transcription, f, ensure_ascii=False)

    hypothesis = transcription["text"]
    reference = utils.get_reference(file, transcription["language"])

    result = utils.compare_transcripts(reference, hypothesis)
    result["language"] = transcription["language"]
    result["file"] = os.path.basename(file)
    result["runtime"] = runtime
    result["options"] = string_options

    return result


def transcribe(file, options):
    model = load_model(options["model_name"])

    whisper_options = options.copy()
    whisper_options.pop("model_name")

    if "language" not in options:
        whisper_options["language"] = get_language(file, options["model_name"])

    audio = load_audio(file)

    result = whisper.transcribe(audio=audio, model=model, **whisper_options)

    result["text"] = result["text"].strip()
    return result


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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model(model_name, device=device)


@cache
def load_audio(file):
    return whisper.load_audio(file)


def whisper_option_combinations():
    # generate a list of all possible combinations of the whisper option values
    for values in product(*whisper_options.values()):
        # generate a dict using the combination values and the original keys
        yield dict(zip(whisper_options.keys(), values))
