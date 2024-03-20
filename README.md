# whisper-pilot
 
[![Build Status](https://github.com/edsu/whisper-pilot/actions/workflows/test.yml/badge.svg)](https://github.com/edsu/whisper-pilot/actions/workflows/test.yml)

This repository contains code for testing OpenAI's Whisper for generating transcripts from audio and video files.

## Run

You'll need to install Poetry to set up the environment:

```
pipx install poetry
```

And then:

```
poetry install
poetry run report.py
```

## Test

To run the unit tests:

```
poetry run pytest
```
