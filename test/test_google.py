import os

import dotenv
from pytest import mark

from transcribe import google

dotenv.load_dotenv()

NO_GOOGLE = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") is None


@mark.skipif(NO_GOOGLE, reason="no Google keys")
def test_transcript():
    result = google.transcribe("test/data/en.wav")
    assert result == {
        "language": "en-us",
        "text": "this is a test for whisper reading in English",
    }


@mark.skipif(NO_GOOGLE, reason="no Google keys")
def test_copy_file():
    assert (
        google.copy_file("test/data/en.wav")
        == "gs://sul-dlss-transcription-edsu-test/en.wav"
    )


def test_convert_to_wav():
    path = google.convert_to_wav("test/data/en.wav")
    assert path.endswith(".wav")
    assert os.path.getsize(path) > 0
