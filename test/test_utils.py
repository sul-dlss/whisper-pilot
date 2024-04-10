import tempfile
from os import path

from transcribe import utils

TEST_DATA = path.join(path.dirname(__file__), "data")


def test_get_data_files():
    files = utils.get_data_files()
    assert len(files) == 11
    assert path.basename(files[0]["media_filename"]) == "bb158br2509_sl.m4a"


def test_compare_transcripts():

    with tempfile.TemporaryDirectory() as output_dir:
        druid = "bb158br2509"

        file_metadata = {
            "druid": druid,
            "media_filename": "foo.mp4",
            "transcript_filename": path.join(TEST_DATA, "en.txt"),
            "run_count": 1,
        }

        whisper_transcript = {
            "language": "en",
            "segments": [{"text": "the quit brown fox jumpsover the crazy dog"}],
        }

        results = utils.compare_transcripts(
            file_metadata, whisper_transcript, "whisper", output_dir
        )

        assert results == {
            "run_id": f"{druid}-whisper-001",
            "druid": druid,
            "file": "foo.mp4",
            "wer": 0.4444444444444444,
            "mer": 0.4444444444444444,
            "wil": 0.6527777777777778,
            "wip": 0.3472222222222222,
            "hits": 5,
            "substitutions": 3,
            "insertions": 0,
            "deletions": 1,
            "language": "en",
            "diff": f"https://sul-dlss.github.io/whisper-pilot/{path.basename(output_dir)}/{druid}-whisper-001.html",
        }

        assert path.isfile(
            path.join(output_dir, f"{druid}-whisper-001.html")
        ), "diff html written"


def test_clean_text():
    assert utils.clean_text(["Makes Lowercase"]) == "makes lowercase"
    assert utils.clean_text(["Strips, punctuation."]) == "strips punctuation"
    assert utils.clean_text(["Removes  spaces"]) == "removes spaces"
    assert (
        utils.clean_text(["Removes    extra      spaces  "]) == "removes extra spaces"
    )
    assert (
        utils.clean_text(["removes\nall\nnewlines", "from\nall\nlines"])
        == "removes all newlines from all lines"
    )
