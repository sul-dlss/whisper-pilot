from os import path

from transcribe import utils

TEST_DATA = path.join(path.dirname(__file__), "data")


def test_get_files():
    files = sorted(utils.get_files(path.join(TEST_DATA, "bags")))
    assert len(files) == 2
    assert path.basename(files[0]) == "bb158br2509_sl.m4a"
    assert path.basename(files[1]) == "gj097zq7635_a_sl.m4a"


def test_get_reference_file():
    files = sorted(utils.get_files(path.join(TEST_DATA, "bags")))
    assert (
        path.basename(utils.get_reference_file(files[0], "en"))
        == "bb158br2509_script.txt"
    )
    assert (
        path.basename(utils.get_reference_file(files[1], "en"))
        == "gj097zq7635_a_sl_script.txt"
    )


def test_compare_transcripts():
    reference = "The quick brown fox jumps over the lazy dog."
    hypothesis = "the quit brown fox jumpsover the crazy dog"
    results = utils.compare_transcripts(reference, hypothesis)
    assert results == {
        "wer": 0.4444444444444444,
        "mer": 0.4444444444444444,
        "wil": 0.6527777777777778,
        "wip": 0.3472222222222222,
        "hits": 5,
        "substitutions": 3,
        "insertions": 0,
        "deletions": 1,
    }


def test_clean_text():
    assert utils.clean_text("Makes Lowercase") == "makes lowercase"
    assert utils.clean_text("Strips, punctuation.") == "strips punctuation"
    assert utils.clean_text("Removes  spaces") == "removes spaces"
    assert utils.clean_text("Removes    extra      spaces  ") == "removes extra spaces"
    assert utils.clean_text("removes\nall\nnewlines") == "removes all newlines"
