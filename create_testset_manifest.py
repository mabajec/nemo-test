"""
Use this script to create manifest file for the TESTSET. 
The manifest is reuired by various NeMo demos and scripts, e.g. speech_to_text_buffered_infer etc.

It is expected that the input folder, i.e. TESTSET folder has the following structure:
TESTSET
    |-  folder1
        |-  audiofile1.wav
        |-  audiofile2.wav
        |-  ...
        |-  audiofilen.wav
    |-  folder2
    |-  ...
    |-  foldern

Author: VITASIS
"""

import os
import json

_CONFIG = {
    "TESTSET_path": "/data/development/TESTSET"
}

def create_test_manifest(folder):
    with open("manifest_testset.txt", "w", encoding="utf-8") as fmanifest:
        for file in os.listdir(folder):
            if file[0] not in (".", "~", "_") and os.path.isdir(os.path.join(folder, file)):
                for ffile in os.listdir(os.path.join(folder, file)):
                    if ffile.split(".")[-1]=="wav" and ffile[0] not in (".", "~"):
                        wavfile = os.path.join(folder, file, ffile)
                        #outfile = os.path.join(parser_output_folder, file, "".join(ffile.split(".")[:-1])+".txt")
                        record = dict(audio_filename=wavfile, text="")
                        fmanifest.write(json.dumps(record, ensure_ascii=False) + '\n')
                        
if __name__ == "__main__":
    create_test_manifest(_CONFIG['TESTSET_path'])
    print("Manifest file created!")