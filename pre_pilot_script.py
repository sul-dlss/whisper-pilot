import whisper,subprocess
import glob, os
import re, json
from pydub import AudioSegment
import numpy as np
from datetime import datetime
import jiwer
import string
import shlex
import csv, sys
import logging

logname = 'pre_pilot.log'
logging.basicConfig(filename=logname,
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.ERROR)

whisper_combinations = [{'beam_size': 5, 'patience': 1.0, 'condition_on_previous_text': True}, {'beam_size': 5, 'patience': 1.0, 'condition_on_previous_text': False}, {'beam_size': 5, 'patience': 2.0, 'condition_on_previous_text': True}, {'beam_size': 5, 'patience': 2.0, 'condition_on_previous_text': False}, {'beam_size': 10, 'patience': 1.0, 'condition_on_previous_text': True}, {'beam_size': 10, 'patience': 1.0, 'condition_on_previous_text': False}, {'beam_size': 10, 'patience': 2.0, 'condition_on_previous_text': True}, {'beam_size': 10, 'patience': 2.0, 'condition_on_previous_text': False}]
preprocessing_combinations = ['afftdn=nr=10:nf=-25:tn=1','afftdn=nr=10:nf=-25:tn=1,volume=4', 'anlmdn,volume=4', 'highpass=200,lowpass=3000,afftdn', 'volume=4', 'speechnorm']
model = whisper.load_model("large")
output_dir = 'outputs'
def run():
	try:
		write_headers = True
		files_with_transcript = get_files()
		for file in files_with_transcript:
			for combination in whisper_combinations:
				start_time = datetime.now()
				whisper_fields = run_whisper(file, combination, file)
				fields = {'file': os.path.basename(file), 'duration': get_runtime(start_time)}
				fields.update(combination)
				fields.update(whisper_fields)
				csv_filename = f"{datetime.now().date()}-spreadsheet.csv"
				write_headers = write_rows(write_headers, fields, csv_filename)
	except Exception as e:
		logging.error(e)


def run_preprocessing():
	try:
		write_headers = True
		files_with_transcript = get_files()
		for file in files_with_transcript:
			for combination in preprocessing_combinations:
				start_time = datetime.now()
				preprocessed_file = os.path.basename(file).rsplit('.', 1)[0] + '_filter_' + re.sub(r'[^A-Za-z0-9]', '', combination) + '.wav'
				p = subprocess.Popen(f"ffmpeg -i {file} -af {combination} {preprocessed_file}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
				whisper_fields = run_whisper(preprocessed_file, {'beam_size': 5, 'patience': 1, 'condition_on_previous_text': True}, file)
				fields = {'file': os.path.basename(file), 'duration': get_runtime(start_time)}
				fields.update({'ffmpeg filter': combination})
				fields.update(whisper_fields)
				os.remove(preprocessed_file)
				csv_filename = f"{datetime.now().date()}-preprocessing-spreadsheet.csv"
				write_headers = write_rows(write_headers, fields, csv_filename)
	except Exception as e:
		logging.error(e)


def write_rows(write_headers, fields, filename):
	with open(filename, mode='a', newline='') as spreadsheet:
		writer = csv.DictWriter(spreadsheet, fieldnames=fields.keys())
		if write_headers:
			writer.writeheader()
			write_headers = False
		writer.writerow(fields)
	return write_headers

def get_files():
	folder = '/home/whisper/Documents/pilot-data/bags/*'
	files_with_transcript = []
	folders = glob.glob(folder + os.path.sep)
	files = [glob.glob('{}data/content/*_sl*.m*'.format(folder)) for folder in folders]
	for folder in files:
		for file in folder:
			if len(glob.glob(f"{file.rsplit('.', 1)[0].replace('_sl', '')}*script.txt")) > 0:
				files_with_transcript.append(file)
				break
	return files_with_transcript

def get_runtime(start_time):
	time_difference = datetime.now() - start_time
	hours, remainder = divmod(time_difference.total_seconds(), 3600)
	minutes, seconds = divmod(remainder, 60)
	return f"{int(hours)}:{int(minutes)}:{int(seconds)}"

def run_whisper(file, options, original_file):
	silences = get_silences(file)
	audio = whisper.load_audio(file)
	if len(silences) > 0 and int(silences[0]['start_silence']) == 0:
		audio_segment = AudioSegment.from_file(file)
		start_time = silences[0]['end_silence'] * 1000
		end_time = (30 + silences[0]['end_silence'])*1000
		clip = audio_segment[start_time:end_time]
		clip.export('cliped_file.wav', format="wav")
		mel = whisper.log_mel_spectrogram('cliped_file.wav', n_mels=128).to(model.device)
		os.remove("cliped_file.wav")
	else:
		audioclip = whisper.pad_or_trim(audio)
		mel = whisper.log_mel_spectrogram(audioclip, n_mels=128).to(model.device)
	_, probs = model.detect_language(mel)
	language = max(probs, key=probs.get)
	result = whisper.transcribe(audio=audio, model=model, word_timestamps=True, language=language, beam_size=options['beam_size'], patience=options['patience'], condition_on_previous_text=options['condition_on_previous_text'])
	if os.path.exists(output_dir) == False:
		os.mkdir(output_dir)
	string_options = "_".join([f"{key}={value}" for key, value in options.items()])
	output_filename = f"{os.path.basename(file)}-{string_options}.json"
	with open(os.path.join(output_dir, output_filename), 'w') as f:
		f.write(json.dumps(result))
	hypothesis = clean_text(result['text'])
	reference_files = glob.glob(f"{original_file.rsplit('.', 1)[0].replace('_sl', '')}*script.txt")
	find_file = list(filter(lambda x: '_{}'.format(language) in x, reference_files))
	reference_file = find_file if len(find_file) > 0 else reference_files
	reference = clean_text(open(reference_file[0]).read())
	output = jiwer.process_words(reference, hypothesis)
	new_dict = { 'wer': output.wer, 'mer': output.mer, 'wil': output.wil,
				 'wip': output.wip, 'hits': output.hits, 'substitutions': output.substitutions,
				 'insertions': output.insertions, 'deletions': output.deletions }
	new_dict['language'] = language
	return new_dict


def clean_text(text):
	text = text.replace("\n", " ").replace("  ", " ")
	text = text.translate(str.maketrans('', '', string.punctuation))
	text = text.lower()
	return text

def get_silences(file):
	file = shlex.quote(file)
	p = subprocess.Popen("ffmpeg -i {} -af 'volumedetect' -vn -sn -dn -f null /dev/null".format(file), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
	detectvolume = p[-1].decode('utf-8')
	maxvolume = ffmpegcontentparse(detectvolume, 'max_volume')
	meanvolume = ffmpegcontentparse(detectvolume, 'mean_volume')
	volume = meanvolume - 1 if meanvolume > -37 else -37
	p2 = subprocess.Popen("ffmpeg -i {} -af silencedetect=n={}dB:d=0.5 -f null -".format(file, volume), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
	startendsilences = []
	for item in p2[-1].decode('utf-8').split("\n"):
		if 'silence_start' in item:
			startendsilences.append({'start_silence': ffmpegcontentparse(item, 'silence_start')})
		elif 'silence_end' in item:
			startendsilences[-1] = startendsilences[-1] | {'end_silence': ffmpegcontentparse(item.split('|')[0], 'silence_end'), 'duration': ffmpegcontentparse(item, 'silence_duration')}

	return startendsilences


def ffmpegcontentparse(content, field):
	lines = content.split('\n')
	correctline = [idx for idx, s in enumerate(lines) if field in s][0]
	get_value = lines[correctline].split(":")[-1]
	value_as_float = float(re.sub(r"[^0-9\-.]", '', get_value, 0, re.MULTILINE).strip())
	return value_as_float

if __name__ == "__main__":
	input_string = sys.argv[1]
	if input_string == 'preprocessing':
		run_preprocessing()
	else:
		run()