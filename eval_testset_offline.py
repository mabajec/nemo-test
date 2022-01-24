# This script is based on speech_to_text_infer.py, available in NeMo/examples/asr

"""
This script serves to evaluate TESTSET files.
Run with arguments. Mandatory args are:
    --asr_model <model_path_and_name>  // currently, only BPE models are supported
    --test_manifest <path_to_a_file_with_test_manifest> // test manifest format: each line shuld have json string {"audio_filename": <path to an audio file>, "text" <baseline for calculating WER. Can be empty string>}
    --model_stride <value> // use 4 for Conformer architecture and 8 for the rest
    --output_path <output_folder_for_transcripts>

Example of run command:   
    python speech_to_text_buffered_infer.py \
        --test_manifest "/data/development/audio_files/manifest_for_buffered_transcript.txt" \ 
        --asr_model "/data/ngc/nemo_models/trained_a40ab7c_conf-bpe.nemo" \
        --output_path "/data/development/NeMo/mydata/output/" --model_stride=4

Author: Marko Bajec
Date: November 2021
"""

import copy
import json
import math
import os
from argparse import ArgumentParser

import torch
from omegaconf import OmegaConf

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.parts.utils.streaming_utils import FrameBatchASR
from nemo.utils import logging
import time


can_gpu = torch.cuda.is_available()
#can_gpu = False


import os
import json

_CONFIG = {
    "TESTSET_path": "/data/development/TESTSET",
    "TESTSET_eval_results_folder": "/data/development/RESULTS",
    "asr_model_path": "/data/ngc/nemo_models/trained_a40ab7c_conf-bpe.nemo",        # if asr_model_path is None, a pretrained NGC model is expected in asr_model_name
    "asr_model_name": "trained_a40ab7c_conf-bpe",
    "batch_size": 32,
    "total_buffer_in_secs": 4.0,    #Length of buffer (chunk + left and right padding) in seconds,
    "chunk_len_in_ms": 1600,        #Chunk length in milliseconds
    "model_stride": 4,              #Model downsampling factor, 8 for Citrinet models and 4 for Conformer models
}


def create_test_manifest(folder, asr_model=""):
    """
    Use this method to create manifest file for the TESTSET. 
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
    """

    with open("manifest_testset.txt", "w", encoding="utf-8") as fmanifest:
        for file in os.listdir(folder):
            if file[0] not in (".", "~", "_") and os.path.isdir(os.path.join(folder, file)):
                for ffile in os.listdir(os.path.join(folder, file)):
                    if ffile.split(".")[-1]=="wav" and ffile[0] not in (".", "~"):
                        wavfile = os.path.join(folder, file, ffile)
                        #outfile = os.path.join(_CONFIG['TESTSET_eval_results_folder'], asr_model, file, "".join(ffile.split(".")[:-1])+".txt")
                        subfolder = file
                        outfile = "".join(ffile.split(".")[:-1])+".txt"
                        record = dict(audio_filename=wavfile, text="", subfolder=subfolder, outfile=outfile)
                        fmanifest.write(json.dumps(record, ensure_ascii=False) + '\n')
                        

def get_wer_feat(mfst, asr, frame_len, tokens_per_chunk, delay, preprocessor_cfg, model_stride_in_secs, device):
    # Create a preprocessor to convert audio samples into raw features,
    # Normalization will be done per buffer in frame_bufferer
    # Do not normalize whatever the model's preprocessor setting is
    preprocessor_cfg.normalize = "None"
    preprocessor = nemo_asr.models.EncDecCTCModelBPE.from_config_dict(preprocessor_cfg)
    preprocessor.to(device)
    hyps = []
    refs = []

    with open(mfst, "r", encoding="utf-8") as mfst_f:
        for l in mfst_f:
            asr.reset()
            row = json.loads(l.strip())
            asr.read_audio_file(row['audio_filepath'], delay, model_stride_in_secs)
            hyp = asr.transcribe(tokens_per_chunk, delay)
            hyps.append(hyp)
            refs.append(row['text'])

    wer = word_error_rate(hypotheses=hyps, references=refs)
    return hyps, refs, wer

def get_transcripts(mfst, asr, frame_len, tokens_per_chunk, delay, preprocessor_cfg, model_stride_in_secs, device):
    preprocessor_cfg.normalize = "None"
    preprocessor = nemo_asr.models.EncDecCTCModelBPE.from_config_dict(preprocessor_cfg)
    preprocessor.to(device)
    hyps = []
    refs = []

    os.makedirs(_CONFIG["TESTSET_eval_results_folder"], exist_ok=True)
    os.makedirs(os.path.join(_CONFIG["TESTSET_eval_results_folder"], _CONFIG["asr_model_name"]), exist_ok=True)
    _path = os.path.join(_CONFIG["TESTSET_eval_results_folder"], _CONFIG["asr_model_name"])

    with open(mfst, "r", encoding="utf-8") as mfst_f:
        for l in mfst_f:
            asr.reset()
            row = json.loads(l.strip())
            asr.read_audio_file(row['audio_filename'], delay, model_stride_in_secs)
            print(f"Transcribing {row['audio_filename']}")
            hyp = asr.transcribe(tokens_per_chunk, delay)

            # save transcript to file
            os.makedirs(os.path.join(_path, row["subfolder"]), exist_ok=True)
            with open(os.path.join(_path, row["subfolder"], row["outfile"]), "w", encoding="utf-8") as fout:
                fout.write(hyp)
            #hyps.append(hyp)
            #refs.append(row['text'])

    #wer = word_error_rate(hypotheses=hyps, references=refs)
    #return hyps, refs, wer


def main():

    # create manifest file
    create_test_manifest(_CONFIG['TESTSET_path'], _CONFIG['asr_model_name'])
    print("Manifest file created!")

    """
    parser = ArgumentParser()
    parser.add_argument(
        "--asr_model", type=str, required=True, help="Path to asr model .nemo file",
    )
    parser.add_argument("--test_manifest", type=str, required=True, help="path to evaluation data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--total_buffer_in_secs",
        type=float,
        default=4.0,
        help="Length of buffer (chunk + left and right padding) in seconds ",
    )
    parser.add_argument("--chunk_len_in_ms", type=int, default=1600, help="Chunk length in milliseconds")
    parser.add_argument("--output_path", type=str, help="path to output file", default=None)
    parser.add_argument(
        "--model_stride",
        type=int,
        default=8,
        help="Model downsampling factor, 8 for Citrinet models and 4 for Conformer models",
    )

    args = parser.parse_args()
    """

    torch.set_grad_enabled(False)
    if _CONFIG['asr_model_path'].endswith('.nemo'):
        logging.info(f"Using local ASR model from {_CONFIG['asr_model_path']}")
        #asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(restore_path=args.asr_model)
        asr_model = nemo_asr.models.ASRModel.restore_from(_CONFIG['asr_model_path'])
    else:
        logging.info(f"Using NGC cloud ASR model {_CONFIG['asr_model_name']}")
        #asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name=args.asr_model)
        asr_model = nemo_asr.models.ASRModel.from_pretrained(_CONFIG['asr_model_name'])

    cfg = copy.deepcopy(asr_model._cfg)
    OmegaConf.set_struct(cfg.preprocessor, False)

    # some changes for streaming scenario
    cfg.preprocessor.dither = 0.0
    cfg.preprocessor.pad_to = 0

    if cfg.preprocessor.normalize != "per_feature":
        logging.error("Only EncDecCTCModelBPE models trained with per_feature normalization are supported currently")

    # Disable config overwriting
    OmegaConf.set_struct(cfg.preprocessor, True)
    asr_model.eval()
    asr_model = asr_model.to(asr_model.device)

    feature_stride = cfg.preprocessor['window_stride']
    model_stride_in_secs = feature_stride * _CONFIG['model_stride']
    total_buffer = _CONFIG['total_buffer_in_secs']

    chunk_len = _CONFIG['chunk_len_in_ms'] / 1000

    tokens_per_chunk = math.ceil(chunk_len / model_stride_in_secs)
    mid_delay = math.ceil((chunk_len + (total_buffer - chunk_len) / 2) / model_stride_in_secs)
    #print(tokens_per_chunk, mid_delay)

    frame_asr = FrameBatchASR(
        asr_model=asr_model, frame_len=chunk_len, total_buffer=_CONFIG['total_buffer_in_secs'], batch_size=_CONFIG['batch_size'],
    )

    """
    hyps, refs, wer = get_wer_feat(
        "manifest_testset.txt",
        frame_asr,
        chunk_len,
        tokens_per_chunk,
        mid_delay,
        cfg.preprocessor,
        model_stride_in_secs,
        asr_model.device,
    )
    """

    start_time = time.time()

    get_transcripts(
        "manifest_testset.txt",
        frame_asr,
        chunk_len,
        tokens_per_chunk,
        mid_delay,
        cfg.preprocessor,
        model_stride_in_secs,
        asr_model.device,
    )

    print("--- %s seconds ---" % (time.time() - start_time))

    #logging.info(f"WER is {round(wer, 2)} when decoded with a delay of {round(mid_delay*model_stride_in_secs, 2)}s")

    
    """
    if args.output_path is not None:

        fname = (
            os.path.splitext(os.path.basename(args.asr_model))[0]
            + "_"
            + os.path.splitext(os.path.basename(args.test_manifest))[0]
            + "_"
            + str(args.chunk_len_in_ms)
            + "_"
            + str(int(total_buffer * 1000))
            + ".json"
        )
        hyp_json = os.path.join(args.output_path, fname)
        os.makedirs(args.output_path, exist_ok=True)
        with open(hyp_json, "w", encoding="utf-8") as out_f:
            for i, hyp in enumerate(hyps):
                record = {
                    "pred_text": hyp,
                    "text": refs[i],
                    "wer": round(word_error_rate(hypotheses=[hyp], references=[refs[i]]) * 100, 2),
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
    """

if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter