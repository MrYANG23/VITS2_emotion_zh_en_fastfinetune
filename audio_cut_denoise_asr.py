import glob
import os
import librosa

import soundfile as sf
from glob import glob
import whisper

import numpy as np

import librosa
import soundfile
import os

import paddle
from paddlespeech.cli.asr import ASRExecutor

asr_executor = ASRExecutor()

# This function is obtained from librosa.
def get_rms(
    y,
    *,
    frame_length=2048,
    hop_length=512,
    pad_mode="constant",
):
    padding = (int(frame_length // 2), int(frame_length // 2))
    y = np.pad(y, padding, mode=pad_mode)

    axis = -1
    # put our new within-frame axis at the end for now
    out_strides = y.strides + tuple([y.strides[axis]])
    # Reduce the shape on the framing axis
    x_shape_trimmed = list(y.shape)
    x_shape_trimmed[axis] -= frame_length - 1
    out_shape = tuple(x_shape_trimmed) + tuple([frame_length])
    xw = np.lib.stride_tricks.as_strided(
        y, shape=out_shape, strides=out_strides
    )
    if axis < 0:
        target_axis = axis - 1
    else:
        target_axis = axis + 1
    xw = np.moveaxis(xw, -1, target_axis)
    # Downsample along the target axis
    slices = [slice(None)] * xw.ndim
    slices[axis] = slice(0, None, hop_length)
    x = xw[tuple(slices)]

    # Calculate power
    power = np.mean(np.abs(x) ** 2, axis=-2, keepdims=True)

    return np.sqrt(power)


class Slicer:
    def __init__(self,
                 sr: int,
                 threshold: float = -40.,
                 min_length: int = 5000,
                 min_interval: int = 300,
                 hop_size: int = 20,
                 max_sil_kept: int = 5000):
        if not min_length >= min_interval >= hop_size:
            raise ValueError('The following condition must be satisfied: min_length >= min_interval >= hop_size')
        if not max_sil_kept >= hop_size:
            raise ValueError('The following condition must be satisfied: max_sil_kept >= hop_size')
        min_interval = sr * min_interval / 1000
        self.threshold = 10 ** (threshold / 20.)
        self.hop_size = round(sr * hop_size / 1000)
        self.win_size = min(round(min_interval), 4 * self.hop_size)
        self.min_length = round(sr * min_length / 1000 / self.hop_size)
        self.min_interval = round(min_interval / self.hop_size)
        self.max_sil_kept = round(sr * max_sil_kept / 1000 / self.hop_size)

    def _apply_slice(self, waveform, begin, end):
        if len(waveform.shape) > 1:
            return waveform[:, begin * self.hop_size: min(waveform.shape[1], end * self.hop_size)]
        else:
            return waveform[begin * self.hop_size: min(waveform.shape[0], end * self.hop_size)]

    # @timeit
    def slice(self, waveform):
        if len(waveform.shape) > 1:
            samples = waveform.mean(axis=0)
        else:
            samples = waveform
        if (samples.shape[0] + self.hop_size - 1) // self.hop_size <= self.min_length:
            return [waveform]
        rms_list = get_rms(y=samples, frame_length=self.win_size, hop_length=self.hop_size).squeeze(0)
        sil_tags = []
        silence_start = None
        clip_start = 0
        for i, rms in enumerate(rms_list):
            # Keep looping while frame is silent.
            if rms < self.threshold:
                # Record start of silent frames.
                if silence_start is None:
                    silence_start = i
                continue
            # Keep looping while frame is not silent and silence start has not been recorded.
            if silence_start is None:
                continue
            # Clear recorded silence start if interval is not enough or clip is too short
            is_leading_silence = silence_start == 0 and i > self.max_sil_kept
            need_slice_middle = i - silence_start >= self.min_interval and i - clip_start >= self.min_length
            if not is_leading_silence and not need_slice_middle:
                silence_start = None
                continue
            # Need slicing. Record the range of silent frames to be removed.
            if i - silence_start <= self.max_sil_kept:
                pos = rms_list[silence_start: i + 1].argmin() + silence_start
                if silence_start == 0:
                    sil_tags.append((0, pos))
                else:
                    sil_tags.append((pos, pos))
                clip_start = pos
            elif i - silence_start <= self.max_sil_kept * 2:
                pos = rms_list[i - self.max_sil_kept: silence_start + self.max_sil_kept + 1].argmin()
                pos += i - self.max_sil_kept
                pos_l = rms_list[silence_start: silence_start + self.max_sil_kept + 1].argmin() + silence_start
                pos_r = rms_list[i - self.max_sil_kept: i + 1].argmin() + i - self.max_sil_kept
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                    clip_start = pos_r
                else:
                    sil_tags.append((min(pos_l, pos), max(pos_r, pos)))
                    clip_start = max(pos_r, pos)
            else:
                pos_l = rms_list[silence_start: silence_start + self.max_sil_kept + 1].argmin() + silence_start
                pos_r = rms_list[i - self.max_sil_kept: i + 1].argmin() + i - self.max_sil_kept
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                else:
                    sil_tags.append((pos_l, pos_r))
                clip_start = pos_r
            silence_start = None
        # Deal with trailing silence.
        total_frames = rms_list.shape[0]
        if silence_start is not None and total_frames - silence_start >= self.min_interval:
            silence_end = min(total_frames, silence_start + self.max_sil_kept)
            pos = rms_list[silence_start: silence_end + 1].argmin() + silence_start
            sil_tags.append((pos, total_frames + 1))
        # Apply and return slices.
        if len(sil_tags) == 0:
            return [waveform]
        else:
            chunks = []
            if sil_tags[0][0] > 0:
                chunks.append(self._apply_slice(waveform, 0, sil_tags[0][0]))
            for i in range(len(sil_tags) - 1):
                chunks.append(self._apply_slice(waveform, sil_tags[i][1], sil_tags[i + 1][0]))
            if sil_tags[-1][1] < total_frames:
                chunks.append(self._apply_slice(waveform, sil_tags[-1][1], total_frames))
            return chunks

def resample_and_save(source, target, sr=16000):
    wav, _ = librosa.load(str(source), sr=sr)
    sf.write(str(target), wav, samplerate=sr, subtype='PCM_16')
    return target


def get_all_wavs(video_dir,audio_dir):
    if not os.path.exists(audio_dir):
        os.mkdir(audio_dir)

    all_dirs=os.listdir(video_dir)
    for per_speaker in all_dirs:

        if not os.path.exists(os.path.join(audio_dir,per_speaker)):
            os.mkdir(os.path.join(audio_dir,per_speaker))


        per_speaker_dir=os.path.join(video_dir,per_speaker)
        all_wavs=os.listdir(per_speaker_dir)
        for per_wav in all_wavs:
            try:
                source_path=os.path.join(per_speaker_dir,per_wav)
                #print('---------------------source_path',source_path)
                target_path=os.path.join(audio_dir,per_speaker,per_wav.split('.')[0]+'.wav')
                resample_and_save(source=source_path,target=target_path)
            except:
                continue




def denoise_wavs_dir(wavs_dir,wavs_out_dir):
    if not os.path.exists(wavs_out_dir):
        os.makedirs(wavs_out_dir)

    all_dirs=os.listdir(wavs_dir)
    for per_dir in tqdm(all_dirs):
        per_dir_path=os.path.join(wavs_dir,per_dir)

        per_out_dir_path=os.path.join(wavs_out_dir,per_dir)
        print('per_dir_path',per_dir_path)
        print('per_out_dir_path',per_out_dir_path)

        if not os.path.exists(per_out_dir_path):
            os.makedirs(per_out_dir_path)

        os.system('python -m denoiser.enhance --dns64 --noisy_dir={} --out_dir={}'.format(per_dir_path,per_out_dir_path))
        os.system('cd {} && rm -rf ./*noisy.wav'.format(per_out_dir_path))
        #exit()
        for per_wav in os.listdir(per_out_dir_path):
            souce_path=os.path.join(per_out_dir_path,per_wav)
            target_path=os.path.join(per_out_dir_path,per_wav.replace('_enhanced',''))
            # print('source_path',souce_path)
            # print('target_path',target_path)
            # exit()
            os.rename(souce_path,target_path)
        #exit()

import os, sys
import requests, signal
from multiprocessing import freeze_support
from multiprocessing import Pool
import time, json
import numpy as np
from pydub import AudioSegment, silence
import random, string
import librosa
from tqdm import tqdm



def cut_per_wav(wav_dir,outdir):
    sound=AudioSegment.from_file(wav_dir)
    sound = sound.set_frame_rate(16000)
    sound = sound.set_channels(1)
    sound = sound.set_sample_width(2)
    silences = silence.detect_silence(sound, min_silence_len=100, silence_thresh=-45, seek_step=5)
    num=len(silences)
    path_name=os.path.basename(wav_dir).split('.')[0]
    start_index=2
    end_index = start_index + np.random.randint(6, 8)
    while end_index <=num-4:
        start=np.random.randint(silences[start_index][0],silences[start_index][1])
        end=np.random.randint(silences[end_index][0],silences[end_index][1])
        final_path=path_name+'_'+str(end_index)
        out_wav = '%s/%s.wav' % (outdir, final_path)
        sound[start:end].export(out_wav, format="wav")
        start_index = end_index
        end_index = start_index + np.random.randint(2, 4)


def cut_per_wav_slice(wav_path,outdir):
    audio, sr = librosa.load(wav_path, sr=None, mono=False)
    duration=len(audio)/sr
    print('------------------duration',duration)
    if duration>15:
        slicer = Slicer(
            sr=16000,
            threshold=-40,
            min_length=5000,
            min_interval=300,
            hop_size=20,
            max_sil_kept=5000
        )
    else:
        slicer = Slicer(
            sr=16000,
            threshold=-40,
            min_length=5000,
            min_interval=300,
            hop_size=20,
            max_sil_kept=5000
        )

    chunks = slicer.slice(audio)
    #print('chunks', chunks)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for i, chunk in enumerate(chunks):
        if len(chunk.shape) > 1:
            chunk = chunk.T
        soundfile.write(os.path.join(outdir, f'%s_%d.wav' % (os.path.basename(wav_path).rsplit('.', maxsplit=1)[0], i)),
                        chunk, sr)




    pass

def cut_audio_in_clent(wav_dirs,out_wavdirs):
    if not os.path.exists(out_wavdirs):
        os.mkdir(out_wavdirs)
    all_wav_names=os.listdir(wav_dirs)
    for per_dir in tqdm(all_wav_names):
        wav_path=os.path.join(wav_dirs,per_dir)
        cut_per_wav(wav_dir=wav_path,outdir=out_wavdirs)



def cut_audio_in_clent_slice(wav_dirs,out_wavdirs):
    if not os.path.exists(out_wavdirs):
        os.mkdir(out_wavdirs)
    all_wav_names=os.listdir(wav_dirs)
    for per_dir in tqdm(all_wav_names):
        wav_path=os.path.join(wav_dirs,per_dir)
        cut_per_wav_slice(wav_path=wav_path,outdir=out_wavdirs)


def audio_cut_wavs_dir(wav_dir,out_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for per_sperker in tqdm(os.listdir(wav_dir)):
        per_sperker_dir=os.path.join(wav_dir,per_sperker)
        per_sperker_out_dir=os.path.join(out_dir,per_sperker)
        cut_audio_in_clent(wav_dirs=per_sperker_dir,out_wavdirs=per_sperker_out_dir)

def audio_cut_wavs_dir_slice(wav_dir,out_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for per_sperker in tqdm(os.listdir(wav_dir)):
        per_sperker_dir=os.path.join(wav_dir,per_sperker)
        per_sperker_out_dir=os.path.join(out_dir,per_sperker)
        if os.path.exists(per_sperker_out_dir):
            print('per_sperker_out_dir',per_sperker_out_dir)
            continue
        cut_audio_in_clent_slice(wav_dirs=per_sperker_dir,out_wavdirs=per_sperker_out_dir)

def text_transcribe(audio_path,model_size='large'):
    model = whisper.load_model(model_size)
    result = model.transcribe(audio=audio_path,fp16=False)
    text = result["text"]
    language = result["language"]
    segments = result["segments"]

    return text,language,segments



def text_transcribe_dirs(audio_dir,asr_outdir,model_size='large'):
    if not os.path.exists(asr_outdir):
        os.makedirs(asr_outdir)

    model = whisper.load_model(model_size)

    all_speakers=os.listdir(audio_dir)
    for per_speaker in  tqdm(all_speakers):
        speaker_dir=os.path.join(audio_dir,per_speaker)
        all_wavs=os.listdir(speaker_dir)
        for per_wav in tqdm(all_wavs):
            audio_path=os.path.join(speaker_dir,per_wav)
            # print('-----------------------audio_path',audio_path)
            # exit()
            result=model.transcribe(audio=audio_path,fp16=False)
            text=result['text']
            #print('-----------------text',text)
            save_path=os.path.join(asr_outdir,per_speaker+'.txt')
            with open(save_path, 'a+') as f:
                result=per_wav+'\t'+text
                f.writelines(result+'\n')



def text_transcribe_dirs_paddle(audio_dir,asr_outdir):
    if not os.path.exists(asr_outdir):
        os.makedirs(asr_outdir)

    model = whisper.load_model('large')

    all_speakers=os.listdir(audio_dir)
    for per_speaker in  tqdm(all_speakers):
        if os.path.exists(os.path.join(asr_outdir,per_speaker+'.txt')):
            continue
        speaker_dir=os.path.join(audio_dir,per_speaker)
        all_wavs=os.listdir(speaker_dir)
        for per_wav in tqdm(all_wavs):
            audio_path=os.path.join(speaker_dir,per_wav)
            if '_en' in audio_path:
                result = model.transcribe(audio=audio_path, fp16=False)
                text = result['text']
            print('-----------------------audio_path',audio_path)
            # exit()
            if '_en' not in audio_path:
                text = asr_executor(
                    model='conformer_wenetspeech',
                    lang='zh',
                    sample_rate=16000,
                    config=None,  # Set `config` and `ckpt_path` to None to use pretrained model.
                    ckpt_path='/data/zll/yanghan/autodl_voice_clone/tts_chinese/epcoh_95.tar.gz',
                    audio_file=audio_path,
                    force_yes=False,
                    device=paddle.get_device())
            # except:
            #     continue
            save_path=os.path.join(asr_outdir,per_speaker+'.txt')
            with open(save_path, 'a+') as f:
                result=per_wav+'\t'+text
                f.writelines(result+'\n')



def auto_resample_denoise_ASRcut(origianl_dir):

    print('start resample')
    get_all_wavs(video_dir=origianl_dir,
                 audio_dir=origianl_dir+'_resample')
    print('resample completed')


    # print('start denoise')
    denoise_wavs_dir(wavs_dir=origianl_dir+'_resample',
                     wavs_out_dir=origianl_dir+'_resample'+'_denoise')
    print('denoise completed')

    print('start cut audio')
    audio_cut_wavs_dir(wav_dir=origianl_dir+'_resample'+'_denoise',
                       out_dir=origianl_dir+'_resample'+'_denoise'+'_audiocut')
    print('audio_cut completed')

    # print('start cut audio')
    # audio_cut_wavs_dir(wav_dir=origianl_dir + '_resample',
    #                    out_dir=origianl_dir + '_resample' + '_denoise' + '_audiocut')
    # print('audio_cut completed')

    print('start asr')
    text_transcribe_dirs(
        audio_dir=origianl_dir+'_resample'+'_denoise'+'_audiocut',
        asr_outdir=origianl_dir+'_resample'+'_denoise'+'_audiocut'+'_asrout')
    print('asr completed')


def auto_resample_denoise_ASRcut_v2(origianl_dir):

    print('start resample')
    get_all_wavs(video_dir=origianl_dir,
                 audio_dir=origianl_dir+'_resample')
    print('resample completed')


    # print('start denoise')
    denoise_wavs_dir(wavs_dir=origianl_dir+'_resample',
                     wavs_out_dir=origianl_dir+'_resample'+'_denoise')
    print('denoise completed')
    #
    print('start cut audio')
    audio_cut_wavs_dir_slice(wav_dir=origianl_dir+'_resample'+'_denoise',
                       out_dir=origianl_dir+'_resample'+'_denoise'+'_audiocut')
    print('audio_cut completed')


    print('start asr')
    text_transcribe_dirs_paddle(
        audio_dir=origianl_dir+'_resample'+'_denoise'+'_audiocut',
        asr_outdir=origianl_dir+'_resample'+'_denoise'+'_audiocut'+'_asrout')
    print('asr completed')





if __name__ == '__main__':
   
    auto_resample_denoise_ASRcut_v2(origianl_dir='/data/zll/yanghan/data/audio/tai_15_speaker_data/tai_26_1024')