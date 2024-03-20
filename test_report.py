import report


def test_get_silences():
    assert report.get_silences("test-data/en.wav") == [
        {"start_silence": 2.70731, "end_silence": 3.22, "duration": 0.512687}
    ]
    assert report.get_silences("test-data/en-with-silence.wav") == [
        {"start_silence": 0.0, "end_silence": 35.1915, "duration": 35.1915}
    ]


def test_get_language():
    assert report.get_language("test-data/en.wav", "small") == "en"
    assert report.get_language("test-data/fr.wav", "small") == "fr"


def test_get_language_with_silence():
    assert report.get_language("test-data/en-with-silence.wav", "small") == "en"
    assert report.get_language("test-data/fr-with-silence.wav", "small") == "fr"
