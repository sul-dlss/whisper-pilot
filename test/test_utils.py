import tempfile
from os import path

from transcribe import utils

TEST_DATA = path.join(path.dirname(__file__), "data")


def test_get_data_files():
    files = utils.get_data_files("data.csv")
    assert len(files) == 11
    assert path.basename(files[0]["media_filename"]) == "bb158br2509_sl.m4a"


def test_compare_transcripts():
    with tempfile.TemporaryDirectory() as output_dir:
        druid = "bb158br2509"

        file_metadata = {
            "druid": druid,
            "media_filename": "en.mp3",
            "transcript_filename": path.join(TEST_DATA, "en.txt"),
            "transcript_language": "en",
            "run_count": 1,
        }

        whisper_transcript = {
            "language": "en",
            "segments": [{"text": "This is a test for cipher reading in English."}],
        }

        results = utils.compare_transcripts(
            file_metadata, whisper_transcript, "whisper", output_dir
        )

        assert results == {
            "run_id": f"{druid}-whisper-001",
            "druid": druid,
            "file": "en.mp3",
            "language": "en",
            "transcript_filename": "en.txt",
            "transcript_language": "en",
            "wer": 0.1111111111111111,
            "mer": 0.1111111111111111,
            "wil": 0.2098765432098766,
            "wip": 0.7901234567901234,
            "hits": 8,
            "substitutions": 1,
            "insertions": 0,
            "deletions": 0,
            "diff": f"https://sul-dlss.github.io/whisper-pilot/{path.basename(output_dir)}/{druid}-whisper-001.html",
        }

        assert path.isfile(
            path.join(output_dir, f"{druid}-whisper-001.html")
        ), "diff html written"


def test_read_txt_reference_file():
    lines = utils.read_reference_file(path.join(TEST_DATA, "en.txt"))
    assert lines == ["This is a test for whisper reading in English."]


def test_read_vtt_reference_file():
    lines = utils.read_reference_file(path.join(TEST_DATA, "en.vtt"))
    assert lines == ["This is a test for whisper reading in English."]


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


def test_strip_rev_formatting():
    assert utils.strip_rev_formatting(["- [interviewer] hi there", "seeya"]) == [
        "hi there",
        "seeya",
    ]


def test_split_sentences():
    assert utils.split_sentences(
        ["This is a test? This is another test... Onwards."]
    ) == ["This is a test?", "This is another test...", "Onwards."]
