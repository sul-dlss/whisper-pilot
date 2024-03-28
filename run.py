#!/usr/bin/env python

import argparse

from transcribe import whisper

parser = argparse.ArgumentParser(
    prog="run", description="Run transcription generation for sample data"
)

parser.add_argument("--preprocessing", "-p", help="Run pre-processing of input files")

args = parser.parse_args()

if args.preprocessing:
    whisper.run_preprocessing()
else:
    whisper.run()
