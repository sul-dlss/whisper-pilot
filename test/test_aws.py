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
        "transcript": "This is a test for whisper reading in English.",
    }
