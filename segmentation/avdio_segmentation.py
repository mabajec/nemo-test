import json
import os
import wget
import numpy as np
import scipy.io.wavfile as wav
import pandas

from tools import prepare_data as prep

# optional
#from plotly import graph_objects as go

_CONFIG = {
		"NEMO_branch": "r1.4.0",
		"ASR_MODEL": "QuartzNet15x5Base-En",
		"organization": {
			"TOOLS_DIR": "tools",
			"WORK_DIR": "work_dir",
			"DATA_DIR": "work_dir/data",
			"OUTPUT_DIR": "work_dir/output"
		},
        "pipeline": {
                "doDownloadUtils": False,
                "doDownloadData": False,
				"oPreprocessData": True
        }
}


if __name__ != "__main__":
        exit()

BRANCH = _CONFIG.get("NEMO_branch")

cur_dir = os.path.abspath(os.getcwd())
TOOLS_DIR = os.path.join(cur_dir, _CONFIG.get("organization").get("TOOLS_DIR"))
WORK_DIR = os.path.join(cur_dir, _CONFIG.get("organization").get("WORK_DIR"))
DATA_DIR = os.path.join(cur_dir, _CONFIG.get("organization").get("DATA_DIR"))


# STEP 1: DOWNLOAD UTILITIES
if _CONFIG.get("pipeline").get("doDownloadUtils"):

	os.makedirs(TOOLS_DIR, exist_ok=True)

	required_files = [
		'prepare_data.py',
        'normalization_helpers.py',
		'run_ctc_segmentation.py',
		'verify_segments.py',
		'cut_audio.py',
		'process_manifests.py',
		'utils.py'
		]
	
	for file in required_files:
		if not os.path.exists(os.path.join(TOOLS_DIR, file)):
			file_path = 'https://raw.githubusercontent.com/NVIDIA/NeMo/' + BRANCH + '/tools/ctc_segmentation/scripts/' + file
			print("\nDownloading file: %s" %file_path)
			wget.download(file_path, TOOLS_DIR)
		elif not os.path.exists(TOOLS_DIR):
			raise ValueError(f'update path to NeMo root directory')

# STEP 2: create data directory and download an audio file
if _CONFIG.get("pipeline").get("doDownloadData"):
     
     os.makedirs(DATA_DIR, exist_ok=True)
     audio_file = 'childrensshortworks019_06acarriersdog_am_128kb.mp3'
     if not os.path.exists(os.path.join(DATA_DIR, audio_file)):
         print('Downloading audio file')
         wget.download('http://archive.org/download/childrens_short_works_vol_019_1310_librivox/' + audio_file, DATA_DIR)

     # text source: http://www.gutenberg.org/cache/epub/24263/pg24263.txt
     text =  """
        A carrier on his way to a market town had occasion to stop at some houses
        by the road side, in the way of his business, leaving his cart and horse
        upon the public road, under the protection of a passenger and a trusty
        dog. Upon his return he missed a led horse, belonging to a gentleman in
        the neighbourhood, which he had tied to the end of the cart, and likewise
        one of the female passengers. On inquiry he was informed that during his
        absence the female, who had been anxious to try the mettle of the pony,
        had mounted it, and that the animal had set off at full speed. The carrier
        expressed much anxiety for the safety of the young woman, casting at the
        same time an expressive look at his dog. Oscar observed his master's eye,
        and aware of its meaning, instantly set off in pursuit of the pony, which
        coming up with soon after, he made a sudden spring, seized the bridle, and
        held the animal fast. Several people having observed the circumstance, and
        the perilous situation of the girl, came to relieve her. Oscar, however,
        notwithstanding their repeated endeavours, would not quit his hold, and
        the pony was actually led into the stable with the dog, till such time as
        the carrier should arrive. Upon the carrier entering the stable, Oscar
        wagged his tail in token of satisfaction, and immediately relinquished the
        bridle to his master.
     """

     with open(os.path.join(DATA_DIR, audio_file.replace('mp3', 'txt')), 'w') as f:
          f.write(text)

# STEP 3: preproces data
if _CONFIG.get("pipeline").get("doPreprocessData"):

	MODEL = _CONFIG.get("ASR_model")
	OUTPUT_DIR = _CONFIG.get("organization").get("OUTPUT_DIR")

	prep.do_prepare_data(
		output_dir=OUTPUT_DIR+"/processed/",
		audio_dir=DATA_DIR,
		additional_split_symbols="",
		use_nemo_normalization=False,
		in_text=DATA_DIR,
		model=MODEL,
		cut_prefix=3
	)


