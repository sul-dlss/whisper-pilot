import os

import dotenv
from pytest import mark

from transcribe import google, utils

dotenv.load_dotenv()

NO_GOOGLE = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") is None


@mark.skipif(NO_GOOGLE, reason="no Google keys")
def test_transcript():
    result = google.transcribe(
        {"media_filename": "test/data/en.wav", "media_language": "en"}
    )
    assert (
        result["results"][0]["alternatives"][0]["transcript"]
        == "this is a test for whisper reading in English"
    )
    assert result["results"][0]["languageCode"] == "en-us"

    # test that the transcript can be turned into text, language (without another request)
    lines, lang = utils.parse_google(result)
    assert lines == ["this is a test for whisper reading in English"]
    assert lang == "en-us"


@mark.skipif(NO_GOOGLE, reason="no Google keys")
def test_transcript_fr():
    result = google.transcribe(
        {"media_filename": "test/data/fr.wav", "media_language": "fr"}
    )
    assert (
        result["results"][0]["alternatives"][0]["transcript"]
        == "il s'agit d'un test de lecture de Whisper en français"
    )
    assert result["results"][0]["languageCode"] == "fr-fr"


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
