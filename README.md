# whisper-pilot
 
[![Build Status](https://github.com/edsu/whisper-pilot/actions/workflows/test.yml/badge.svg)](https://github.com/edsu/whisper-pilot/actions/workflows/test.yml)

This repository contains code for testing OpenAI's Whisper for generating transcripts from audio and video files.

## Run

You'll probably want a virtual environment:


```
$ python -m venv env
$ source env/bin/activate
```

Then you'll need to install dependencies:

```
$ pip install -r requirements.txt
```

Then you can run the report:

```
$ ./report.py
```

## Test

To run the unit tests:

```
$ pytest
```
