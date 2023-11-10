import sys
TTS_PATH = "TTS/"

# add libraries into environment
sys.path.append(TTS_PATH) # set this if TTS is not installed globally

import os
import string
import time
import argparse
import json

import numpy as np
import IPython
from IPython.display import Audio
import torch
from TTS.tts.utils.synthesis import synthesis
#from TTS.tts.utils.text.symbols import make_symbols, phonemes, symbols

try:
  from TTS.utils.audio import AudioProcessor
except:
  from TTS.utils.audio import AudioProcessor


from TTS.tts.models import setup_model
from TTS.config import load_config
from TTS.tts.models.vits import *

CONFIG_SE_PATH = "model_dir/config_se.json"
CHECKPOINT_SE_PATH = "model_dir/model_se.pth.tar"

from TTS.tts.utils.speakers import SpeakerManager
from pydub import AudioSegment
import librosa
from tqdm import tqdm


USE_CUDA = torch.cuda.is_available()

SE_speaker_manager = SpeakerManager(encoder_model_path=CHECKPOINT_SE_PATH, encoder_config_path=CONFIG_SE_PATH, use_cuda=USE_CUDA)

import librosa
import soundfile as sf
from tqdm import tqdm


def resample_file(filename,output_sr):
    y, sr = librosa.load(filename, sr=output_sr)
    sf.write(filename, y, sr)




def get_dataset_embedding_save(filelist,speaker_encoder,save_path):
    """

    :rtype: object
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    speaker_embedding={}
    save_speaker={}
    with open(filelist,'r') as f:
        results=f.readlines()
        for per_line in tqdm(results):
            #try:
            path,sid,_=per_line.strip().split('|')
            #print('------------per_line',per_line)
            speaker_embedding[str(sid)]=[]
            sid_embdding=torch.FloatTensor(speaker_encoder.compute_d_vector_from_clip(path)).unsqueeze(0)
            #print('sid_embedding.',sid_embdding.shape)
            speaker_embedding[str(sid)].append(sid_embdding)
            # except:
            #     continue

    for key,vaule in speaker_embedding.items():
        all_sid_embeding=speaker_embedding[key]
        num_utterce=len(all_sid_embeding)

        sid_embeding =sum(all_sid_embeding)/num_utterce
        #print('----------------------key',key)
        np.save(os.path.join(save_path,str(key)),sid_embeding)





def get_dataset_embedding_mix_train_save(filelist1,filelist2,speaker_encoder,save_path):
    """
    :rtype: object
    """

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    speaker_embedding={}
    save_speaker={}
    total_results=[]

    with open(filelist1,'r') as f1:
        results1=f1.readlines()
        total_results.extend(results1)
    with open(filelist2,'r') as f2:
        results2=f2.readlines()
        total_results.extend(results2)


    for per_line in tqdm(total_results):
        #try:
        path,sid,_=per_line.strip().split('|')
        #print('------------per_line',per_line)
        speaker_embedding[str(sid)]=[]
        sid_embdding=torch.FloatTensor(speaker_encoder.compute_d_vector_from_clip(path)).unsqueeze(0)
        #print('sid_embedding.',sid_embdding.shape)
        speaker_embedding[str(sid)].append(sid_embdding)
        # except:
            #     continue

    for key,vaule in speaker_embedding.items():
        all_sid_embeding=speaker_embedding[key]
        num_utterce=len(all_sid_embeding)

        sid_embeding =sum(all_sid_embeding)/num_utterce
        #print('----------------------key',key)
        np.save(os.path.join(save_path,str(key)),sid_embeding)


def get_dataset_embedding_persave(filelist,speaker_encoder,save_path):
    """

    :rtype: object
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    speaker_embedding={}
    save_speaker={}
    with open(filelist,'r') as f:
        results=f.readlines()
        for per_line in tqdm(results):
            #try:
            path,sid,_=per_line.strip().split('|')
            #print('------------per_line',per_line)
            speaker_embedding[str(sid)]=[]
            sid_embdding=torch.FloatTensor(speaker_encoder.compute_d_vector_from_clip(path)).unsqueeze(0)
            #print('sid_embedding.',sid_embdding.shape)
            speaker_embedding[str(sid)].append(sid_embdding)
            # except:
            #     continue

    for key,vaule in speaker_embedding.items():
        all_sid_embeding=speaker_embedding[key]
        num_utterce=len(all_sid_embeding)

        sid_embeding =sum(all_sid_embeding)/num_utterce
        #print('----------------------key',key)
        np.save(os.path.join(save_path,str(key)),sid_embeding)





def get_dataset_embedding(filelist,speaker_encoder):
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--name", help="Description of arg1")

    args = parser.parse_args()
    #
    # print(args.arg1)
    # print(args.arg2)
    
    """
    :rtype: object
    """
    save_path = '../tts_chinese/speaker_embedding_dir/{}'.format(args.name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    speaker_embedding={}
    save_speaker={}
    with open(filelist,'r') as f:
        results=f.readlines()
        for per_line in tqdm(results):
            #try:
            path,sid,_=per_line.strip().split('|')
            #print('------------per_line',per_line)
            speaker_embedding[str(sid)]=[]
            sid_embdding=torch.FloatTensor(speaker_encoder.compute_d_vector_from_clip(path)).unsqueeze(0)
            #print('sid_embedding.',sid_embdding.shape)
            speaker_embedding[str(sid)].append(sid_embdding)
            # except:
            #     continue

    for key,vaule in speaker_embedding.items():
        all_sid_embeding=speaker_embedding[key]
        num_utterce=len(all_sid_embeding)
        sid_embeding =sum(all_sid_embeding)/num_utterce
        #print('----------------------key',key)
        np.save(os.path.join(save_path,str(key)),sid_embeding)


def get_per_dataset_embedding(filelist,speaker_encoder,save_path):
    if not os.path.join(save_path):
        os.makedirs(save_path)
    with open(filelist,'r') as f:
        results=f.readlines()
        for per_line in tqdm(results):
            path,sid,_=per_line.strip().split('|')
            new_speker_dir=os.path.join(save_path,sid)
            if not os.path.exists(new_speker_dir):
                os.makedirs(new_speker_dir)
            name=path.split('/')[-1].split('.')[0]
            #print('--------------------name',name)
            embedding = torch.FloatTensor(speaker_encoder.compute_d_vector_from_clip(path)).unsqueeze(0)
            #print('save_path',os.path.join(new_speker_dir,name+'.npy'))
            #exit()
            np.save(os.path.join(new_speker_dir, name + '.npy'), embedding)



def get_wavdir_embedding(wav_dir,speaker_encoder,save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    all_wavs=os.listdir(wav_dir)

    name=wav_dir.split('/')[-1]
    all_embeddings=[]
    for per_wav_path in tqdm(all_wavs):
        resample_file(filename=os.path.join(wav_dir,per_wav_path),output_sr=16000)
        embedding=torch.FloatTensor(speaker_encoder.compute_d_vector_from_clip(os.path.join(wav_dir,per_wav_path))).unsqueeze(0)
        all_embeddings.append(embedding)
    nums=len(all_embeddings)
    sid_embedding=sum(all_embeddings)/nums
    np.save(os.path.join(save_path,name+'.npy'),sid_embedding)


def get_singel_wav_embedding(wav_path,speaker_encoder,save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    embedding = torch.FloatTensor(
        speaker_encoder.compute_d_vector_from_clip(wav_path)).unsqueeze(0)
    np.save(os.path.join(save_path, wav_path.split('/')[-1].split('.')[0] + '.npy'), embedding)




if __name__ == '__main__':
    # get_dataset_embedding(
    #     filelist='/data/zll/yanghan/autodl_voice_clone/tts_chinese/filelists/finetune_zh_en_mix_train_v2.txt',
    #     speaker_encoder=SE_speaker_manager,
    #    )
    #
    get_dataset_embedding_save(
        filelist='/data/zll/yanghan/autodl_voice_clone/tts_chinese/filelists/test_1009_nocai_add_ori.txt',
        speaker_encoder=SE_speaker_manager,
        save_path='/data/zll/yanghan/autodl_voice_clone/tts_chinese/speaker_embedding_dir/test_1009_nocai')

    # get_dataset_embedding_mix_train_save(filelist1='/data/zll/yanghan/autodl_voice_clone/tts_chinese/filelists/0907_5xunfei_xiaoyun_train.txt',
    #                                      filelist2='/data/zll/yanghan/autodl_voice_clone/tts_chinese/filelists/1000_aishell_1000_vctk.txt',
    #                                      speaker_encoder=SE_speaker_manager,
    #                                      save_path='/data/zll/yanghan/autodl_voice_clone/tts_chinese/speaker_embedding_dir/0907_5xunfei_xiaoyun_2000')