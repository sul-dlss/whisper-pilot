from os import path

import torch

from transcribe import whisper

MODEL_SIZE = "large" if torch.cuda.is_available() else "small"
TEST_DATA = path.join(path.dirname(__file__), "data")


def test_get_silences():
    assert whisper.get_silences(path.join(TEST_DATA, "en.wav")) == [
        {"start_silence": 2.70731, "end_silence": 3.22, "duration": 0.512687}
    ]
    assert whisper.get_silences(path.join(TEST_DATA, "en-with-silence.wav")) == [
        {"start_silence": 0.0, "end_silence": 35.1915, "duration": 35.1915}
    ]


def test_get_language():
    assert whisper.get_language(path.join(TEST_DATA, "en.wav"), MODEL_SIZE) == "en"
    assert whisper.get_language(path.join(TEST_DATA, "fr.wav"), MODEL_SIZE) == "fr"


def test_get_language_with_silence():
    assert (
        whisper.get_language(path.join(TEST_DATA, "en-with-silence.wav"), MODEL_SIZE)
        == "en"
    )
    assert (
        whisper.get_language(path.join(TEST_DATA, "fr-with-silence.wav"), MODEL_SIZE)
        == "fr"
    )
