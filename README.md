# whisper-pilot
 
[![Build Status](https://github.com/sul-dlss/whisper-pilot/actions/workflows/test.yml/badge.svg)](https://github.com/sul-dlss/whisper-pilot/actions/workflows/test.yml)

This repository contains code for testing OpenAI's Whisper for generating transcripts from audio and video files.

## Data

The data used in this analysis was determined ahead of time in this spreadsheet, which has a snapshot included in this repository as `data.csv`:

https://docs.google.com/spreadsheets/d/1sgcxy0eNwWTn1LeMVH8TDJ6J8qL8iIGfZ25t4nmYqyQ/edit#gid=0

The items were exported as BagIt directories from SDR preservation using the [SDRGET](https://consul.stanford.edu/pages/viewpage.action?pageId=1646529897) process. The total amount of data is 596 GB. This includes the preservation masters, and service copies. Depending on the available storage you may only want to copy the service copies, but you'll want to preserve the directory structure of the bags.

So assuming SDR-GET exported the bags to `/path/to/export` and you want rsync just the low service copies to `example.stanford.edu` you can:

```
rsync -rvhL --times /path/to/export user@example.stanford.edu:pilot-data
```

## Setup

Create a virtual environment:

```
$ python -m venv env
$ source env/bin/activate
```

Install dependencies:

```
$ pip install -r requirements.txt
```

## Run

Then you can run the report:

```
$ ./run.py /path/to/bags
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

## Analysis

There is a Jupyter [Notebook](https://github.com/sul-dlss/whisper-pilot/blob/main/Notebook.ipynb) that contains some analysis of the results.

```
$ jupyter lab Notebook.ipynb
```
