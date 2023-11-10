import os
import random
import time

import numpy as np
import torch
import torch.utils.data

import commons
from mel_processing import (mel_spectrogram_torch, spec_to_mel_torch,
                            spectrogram_torch)
from text import cleaned_text_to_sequence_mix,text_to_sequence_mix
from utils import load_filepaths_and_text, load_wav_to_torch


class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
    1) loads audio, speaker_id, text pairs
    2) normalizes text and converts them to sequences of integers
    3) computes spectrograms from audio files.
    """
    def __init__(self, audiopaths_sid_text, hparams):
        self.hparams = hparams
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.sampling_rate = hparams.sampling_rate
        self.mix_train_filelist=hparams.mix_train_files
        self.mix_train_num_per_batch=hparams.mix_train_num_per_batch

        self.use_mel_spec_posterior = getattr(
            hparams, "use_mel_posterior_encoder", False
        )
        if self.use_mel_spec_posterior:
            self.n_mel_channels = getattr(hparams, "n_mel_channels", 80)
        self.cleaned_text = getattr(hparams, "cleaned_text", False)

        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 190)
        self.min_audio_len = getattr(hparams, "min_audio_len", 8192)
        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)
        self._filter()

        if self.mix_train_filelist!="":
            self.mix_audiopaths_sid_text=load_filepaths_and_text(self.mix_train_filelist)
            random.seed(1234)
            random.shuffle(self.mix_audiopaths_sid_text)
            self._mix_filter()
        else:
            self.mix_audiopaths_sid_text=[]
        self.lengths=len(audiopaths_sid_text)+len(self.mix_audiopaths_sid_text)

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length
        audiopaths_sid_text_new = []
        lengths = []
        for audiopath, sid, text,lang in self.audiopaths_sid_text:
            if not os.path.isfile(audiopath):
                continue
            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                audiopaths_sid_text_new.append([audiopath, sid, text,lang])
                length = os.path.getsize(audiopath) // (2 * self.hop_length)
                if length < self.min_audio_len // self.hop_length:
                    continue
                lengths.append(length)
        self.audiopaths_sid_text = audiopaths_sid_text_new
        #self.lengths = lengths
        print('len_base_filelist',
            len(self.lengths)
        )  # if we use large corpus dataset, we can check how much time it takes.

    def _mix_filter(self):

        """
               Filter text & store spec lengths
               """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length
        mix_audiopaths_sid_text_new = []
        lengths = []
        for audiopath, sid, text,lang in self.mix_audiopaths_sid_text:
            if not os.path.isfile(audiopath):
                continue
            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                mix_audiopaths_sid_text_new.append([audiopath, sid, text,lang])
                length = os.path.getsize(audiopath) // (2 * self.hop_length)
                if length < self.min_audio_len // self.hop_length:
                    continue
                lengths.append(length)
        self.mix_audiopaths_sid_text = mix_audiopaths_sid_text_new
        self.mix_lengths = lengths
        print(
            'len_mix_filelist',len(self.mix_lengths)
        )  # if we use large corpus dataset, we can check how much time it takes.




    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        # separate filename, speaker_id and text
        audiopath, sid, text,lang = (
            audiopath_sid_text[0],
            audiopath_sid_text[1],
            audiopath_sid_text[2],
            audiopath_sid_text[3],
        )

        text = self.get_text(text)
        spec, wav = self.get_audio(audiopath)
        sid = self.get_sid(sid)
        emo=torch.FloatTensor(np.load(audiopath+"emo.npy"))
        emb=torch.FloatTensor(np.load(audiopath+"emb.npy"))
        return (text, spec, wav, sid,lang,emo,emb)

    def get_audio(self, filename):
        # TODO : if linear spec exists convert to mel from existing linear spec
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                "{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate
                )
            )
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if self.use_mel_spec_posterior:
            spec_filename = spec_filename.replace(".spec.pt", ".mel.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            if self.use_mel_spec_posterior:
                """TODO : (need verification)
                if linear spec exists convert to
                mel from existing linear spec (uncomment below lines)"""
                # if os.path.exists(filename.replace(".wav", ".spec.pt")):
                #     # spec, n_fft, num_mels, sampling_rate, fmin, fmax
                #     spec = spec_to_mel_torch(
                #         torch.load(filename.replace(".wav", ".spec.pt")),
                #         self.filter_length, self.n_mel_channels, self.sampling_rate,
                #         self.hparams.mel_fmin, self.hparams.mel_fmax)
                spec = mel_spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.n_mel_channels,
                    self.sampling_rate,
                    self.hop_length,
                    self.win_length,
                    self.hparams.mel_fmin,
                    self.hparams.mel_fmax,
                    center=False,
                )
            else:
                spec = spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.sampling_rate,
                    self.hop_length,
                    self.win_length,
                    center=False,
                )
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)
        return spec, audio_norm

    def get_text(self, text):
        if self.cleaned_text:
            text_norm = cleaned_text_to_sequence_mix(text)
        else:
            text_norm = text_to_sequence_mix(text, self.text_cleaners)
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def get_sid(self, sid):
        sid = torch.LongTensor([int(sid)])
        return sid

    def __getitem__(self, index):
        if self.mix_train_filelist != "":
            return self.get_audio_text_speaker_pair((self.audiopaths_sid_text+self.mix_audiopaths_sid_text)[index])
        else:
            return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):

        if self.mix_train_filelist != "":
            return len(self.audiopaths_sid_text+self.mix_audiopaths_sid_text)
        else:
            return len(self.audiopaths_sid_text)


class TextAudioSpeakerCollate:
    """Zero-pads model inputs and targets"""

    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]), dim=0, descending=True
        )

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        emo = torch.FloatTensor(len(batch), 1024)
        emb = torch.FloatTensor(len(batch),512)
        language_padded=torch.LongTensor(len(batch),max_text_len)


        text_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        emo.zero_()
        emb.zero_()
        language_padded.zero_()





        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, : text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, : wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            sid[i] = row[3]

            language=row[4]
            language_padded[i,:language.size(0)]=language

            emo[i,:]=row[5]
            emb[i,:]=row[6]


        if self.return_ids:
            return (
                text_padded,
                text_lengths,
                spec_padded,
                spec_lengths,
                wav_padded,
                wav_lengths,
                sid,
                language_padded,
                emo,
                emb,
                ids_sorted_decreasing,
            )
        return (
            text_padded,
            text_lengths,
            spec_padded,
            spec_lengths,
            wav_padded,
            wav_lengths,
            sid,
            language_padded,
            emo,
            emb
        )


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        boundaries,
        num_replicas=None,
        rank=None,
        shuffle=True,
    ):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas
        self.mix_train_filelist=dataset.mix_audiopaths_sid_text
        self.num_per_batch=dataset.mix_train_num_per_batch

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (
                total_batch_size - (len_bucket % total_batch_size)
            ) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = (
                ids_bucket
                + ids_bucket * (rem // len_bucket)
                + ids_bucket[: (rem % len_bucket)]
            )

            # subsample
            ids_bucket = ids_bucket[self.rank :: self.num_replicas]

            if len(self.mix_train_filelist) != 0:
                # batching
                for j in range(len(ids_bucket) // self.batch_size):
                    batch = [
                        bucket[idx]
                        for idx in ids_bucket[
                            j * self.batch_size : (j + 1) * self.batch_size
                        ]
                    ]

                    ########################mix_train##################
                    mix_id=[i for i in range(self.lengths-len(self.mix_train_filelist),self.lengths)]
                    finetune_id = torch.randperm(len(mix_id), generator=g).tolist()
                    random_finetune_id = random.sample(finetune_id, self.num_per_batch)
                    ########################指定固定比例################
                    batch.extend(mix_id[i] for i in random_finetune_id)
                    ########################mix_train##################

                    batches.append(batch)
            else:
                # batching
                for j in range(len(ids_bucket) // self.batch_size):
                    batch = [
                        bucket[idx]
                        for idx in ids_bucket[
                                   j * self.batch_size: (j + 1) * self.batch_size
                                   ]
                    ]
                    batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size