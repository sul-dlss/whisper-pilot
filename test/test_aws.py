from os import environ, path

import dotenv
from pytest import mark

from transcribe import aws

dotenv.load_dotenv()

NO_AWS = environ.get("AWS_ACCESS_KEY_ID") is None
TEST_DATA = path.join(path.dirname(__file__), "data")


@mark.skipif(NO_AWS, reason="no AWS keys")
def test_transcript():
    result = aws.transcribe({"media_filename": path.join(TEST_DATA, "en.wav")})
    assert (
        result["results"]["transcripts"][0]["transcript"]
        == "This is a test for whisper reading in English."
    )
    assert result["results"]["language_code"] == "en-US"


@mark.skipif(NO_AWS, reason="no AWS keys")
def test_transcript_with_silence():
    result = aws.transcribe(
        {"media_filename": path.join(TEST_DATA, "en-with-silence.wav")}
    )
    assert (
        result["results"]["transcripts"][0]["transcript"]
        == "This is a test for whisper reading in English."
    )
    assert result["results"]["language_code"] == "en-US"


@mark.skipif(NO_AWS, reason="no AWS keys")
def test_transcript_fr():
    result = aws.transcribe({"media_filename": path.join(TEST_DATA, "fr.wav")})
    assert (
        result["results"]["transcripts"][0]["transcript"]
        == "Il s'agit d'un test de lecture de Whisper en français."
    )
    assert result["results"]["language_code"] == "fr-FR"
