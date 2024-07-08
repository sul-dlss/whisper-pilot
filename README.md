# whisper-pilot
 
[![Build Status](https://github.com/sul-dlss/whisper-pilot/actions/workflows/test.yml/badge.svg)](https://github.com/sul-dlss/whisper-pilot/actions/workflows/test.yml)

This repository contains code for testing OpenAI's Whisper for generating transcripts from audio and video files, and comparing results with AWS Transcribe and Google Speech APIs.

## Data

The data used in this analysis was determined ahead of time in this spreadsheet, which has a snapshot included in this repository as `sdr-data.csv`:

https://docs.google.com/spreadsheets/d/1sgcxy0eNwWTn1LeMVH8TDJ6J8qL8iIGfZ25t4nmYqyQ/edit#gid=0

The items were exported as BagIt directories from SDR preservation using the [SDRGET](https://consul.stanford.edu/pages/viewpage.action?pageId=1646529897) process. The total amount of data is 596 GB. This includes the preservation masters, and service copies. Depending on the available storage you may only want to copy the service copies, but you'll want to preserve the directory structure of the bags.

So assuming SDR-GET exported the bags to `/path/to/export` and you want rsync just the low service copies to `example.stanford.edu` you can:

```
rsync -rvhL --times --include "*/" --include "*.mp4" --include "*.m4a" --include "*.txt" --exclude "*" /path/to/export user@example.stanford.edu:pilot-data
```

The bags should be made available in a `data` directory that you create in the same directory you've cloned this repository to. Alternatively you can symlink the location to `data`

## Manifest

The specific media files and the transcripts that will be used as the gold standard for comparison are in the "manifest" `data.csv`. This file is what determines which files are transcribed, and where the transcription to compare against is. You will notice that the file paths assume they are relative to the `data` directory.

## Whisper Options

The whisper options that are perturbed as part of the run are located in the whisper module:

https://github.com/sul-dlss/whisper-pilot/blob/83292dc8f32bc30a003d0e71362ad12733f66473/transcribe/whisper.py#L27-L33

I guess these could have been command line options or a separate configuration file, but we knew what we wanted to test. This is where to make adjustments if you do want to test additional Whisper options.

## Setup

Create or link your data directory:

```
$ ln -s /path/to/exported/data data
```

Create a virtual environment:

```
$ python -m venv env
$ source env/bin/activate
```

Install dependencies:

```
$ pip install -r requirements.txt
```

To run the AWS and Google tests you'll need to:

```
$ cp env-example .env
```

And then edit it to add the relevant keys and other platform specific configuration.

## Run

Then you can run the report:

```
$ ./run.py
```

If you just want to run one of the report types you can, for example only run the AWS jobs:

```
$ ./run --only aws
```

## Test

To run the unit tests you should:

```
$ pytest
```

## Analysis

There are some Jupyter notebooks in the `notebooks` directory which you can view here on Github.

* [Caption Providers](https://github.com/sul-dlss/whisper-pilot/blob/main/notebooks/caption-providers.ipynb): an analysis of Word Error Rates for Whisper, Google Speech and Amazon Transcribe.
* [On Prem Estimate](https://github.com/sul-dlss/whisper-pilot/blob/main/notebooks/on-prem-estimate.ipynb): an estimate of how long it will take to run our backlog through Whisper using hardware similar to the RDS GPU work station.
* [Whisper Options](https://github.com/sul-dlss/whisper-pilot/blob/main/notebooks/whisper-options.ipynb) examining the effects of adjusting several Whisper options.

If you want to interact with them you'll need to run Jupyter Lab which was installed with the dependencies:

```
$ jupyter lab
```
