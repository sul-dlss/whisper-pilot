from os import environ, path

import dotenv
from pytest import mark

from transcribe import aws

dotenv.load_dotenv()

NO_AWS = environ.get("AWS_ACCESS_KEY_ID") is None
TEST_DATA = path.join(path.dirname(__file__), "data")


@mark.skipif(NO_AWS, reason="no AWS keys")
def test_transcript():
    result = aws.transcribe(path.join(TEST_DATA, "en.wav"))
    assert result == {
        "language": "en-US",
        "text": "This is a test for whisper reading in English.",
    }


@mark.skipif(NO_AWS, reason="no AWS keys")
def test_transcript_with_silence():
    result = aws.transcribe(path.join(TEST_DATA, "en-with-silence.wav"))
    assert result == {
        "language": "en-US",
        "text": "This is a test for whisper reading in English.",
    }


@mark.skipif(NO_AWS, reason="no AWS keys")
def test_transcript_fr():
    result = aws.transcribe(path.join(TEST_DATA, "fr.wav"))
    assert result == {
        "language": "fr-FR",
        "text": "Il s'agit d'un test de lecture de Whisper en français.",
    }
