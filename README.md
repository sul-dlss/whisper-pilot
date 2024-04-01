# whisper-pilot
 
[![Build Status](https://github.com/sul-dlss/whisper-pilot/actions/workflows/test.yml/badge.svg)](https://github.com/sul-dlss/whisper-pilot/actions/workflows/test.yml)

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
$ ./run.py
```

## Test

To run the unit tests you should:

```
$ pytest
```

If you want to run the AWS and Google tests you'll need to:

```
$ cp env-example .env
```

And then edit it to add the relevant keys and other platform specific configuration.
