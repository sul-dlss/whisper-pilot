import os

import dotenv
from pytest import mark

from transcribe import google

dotenv.load_dotenv()

NO_GOOGLE = os.environ.get("GOOGLE") is None


@mark.skipif(NO_GOOGLE, reason="no Google keys")
def test_transcript():
    result = google.transcribe("test-data/en.wav")
    assert result == {
        "language": "en-US",
        "transcript": "This is a test for whisper reading in English.",
    }
