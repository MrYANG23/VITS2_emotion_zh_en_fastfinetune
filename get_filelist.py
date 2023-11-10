import os
import sys
import numpy as np
from pypinyin import Style
from pypinyin.contrib.neutral_tone import NeutralToneWith5Mixin
from pypinyin.converter import DefaultConverter
from pypinyin.core import Pinyin, load_phrases_dict
from scipy.io import wavfile
import librosa
import soundfile as sf
import datetime
import torch
import utils
from tn.chinese.normalizer import Normalizer
normalizer = Normalizer()

from text import cleaned_text_to_sequence_mix
from text.cleaners import mix_cleaners

from tqdm import tqdm
def resample_and_save(source, target, sr=16000):
    wav, _ = librosa.load(str(source), sr=sr)
    sf.write(str(target), wav, samplerate=sr, subtype='PCM_16')
    return target


import pickle
import json

def get_multispeaker_train_test_filelist(txt_dir,
                                     wav_dir,
                                    filelist_name):

    all_result_list = []
    all_txts = os.listdir(txt_dir)
    for per_txt in tqdm(all_txts):
        txt_path = os.path.join(txt_dir, per_txt)
        with open(txt_path, 'r') as f:
            results = f.readlines()
            for per_line in tqdm(results):
                if len(per_line.strip().split('\t')) != 2:
                    continue
                name, text = per_line.strip().split('\t')
                source_dir = os.path.join(wav_dir, per_txt.split('.')[0])
                source_path = os.path.join(source_dir, name)
                try:
                    print('source_path', source_path)
                    phonemes,lang=mix_cleaners(text)
                    print('phonemes',phonemes)
                    path = source_path + '|' + per_txt.rstrip('.txt') + '|' + '#'.join(phonemes)+'|'+'#'.join([str(i) for i in lang])
                    all_result_list.append(path)
                except:
                    continue


    train_list = all_result_list
    test_list = all_result_list[len(all_result_list) - int(0.1*len(all_result_list)):]
    print('-------------------len(train_list)',len(train_list))
    finetune_train_txt = 'filelists/{}_train.txt'.format(filelist_name)
    finetune_test_txt = 'filelists/{}_val.txt'.format(filelist_name)

    with open(finetune_train_txt, 'w') as f1:
        for per_train_line in train_list:
            f1.writelines(per_train_line + '\n')
    with open(finetune_test_txt, 'w') as f1:
        for per_test_line in test_list:
            f1.writelines(per_test_line + '\n')





if __name__ == '__main__':
    get_multispeaker_train_test_filelist(
        txt_dir='/data/zll/yanghan/data/audio/tai_15_speaker_data/tai_40_1027_audiocut_asrout',
        wav_dir='/data/zll/yanghan/data/audio/tai_15_speaker_data/tai_40_1027_audiocut',
        filelist_name='tai_40_1027')