import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from tokenize import Number
from turtle import shape

import librosa
import numpy as np
import sox
import torch
from torch.utils.data import Dataset, Sampler, DistributedSampler, DataLoader
import torchaudio

from deepspeech_pytorch.configs.train_config import SpectConfig, AugmentationConfig
from deepspeech_pytorch.loader.spec_augment import spec_augment

import pytorch_lightning as pl
from hydra.utils import to_absolute_path

from deepspeech_pytorch.configs.train_config import DataConfig
from deepspeech_pytorch.loader.data_loader import SpectrogramDataset, DSRandomSampler, AudioDataLoader, \
    DSElasticDistributedSampler
    
from dataclasses import dataclass, field

torchaudio.set_audio_backend("sox_io")

def load_audio(path):
    sound, sample_rate = torchaudio.load(path)
    if sound.shape[0] == 1:
        sound = sound.squeeze()
    else:
        sound = sound.mean(axis=0)  # multiple channels, average
    return sound.numpy()

class WavDataset(Dataset):
    def __init__(self,
                 input_path: str,
                 labels: list,
                 sample_length: int = None,
                 normalize: bool = False,
                 aug_cfg: AugmentationConfig = None):
        """
        Dataset that loads tensors via a csv containing file paths to audio files and transcripts separated by
        a comma. Each new line is a different sample. Example below:

        /path/to/audio.wav,/path/to/audio.txt
        ...
        You can also pass the directory of dataset.
        :param audio_conf: Config containing the sample rate, window and the window length/stride in seconds
        :param input_path: Path to input.
        :param labels: List containing all the possible characters to map to
        :param normalize: Apply standard mean and deviation normalization to audio tensor
        :param augmentation_conf(Optional): Config containing the augmentation parameters
        """
        self.ids = self._parse_input(input_path)
        self.size = len(self.ids)
        self.labels_map = dict([(labels[i], i) for i in range(len(labels))])
        self.sample_length = sample_length
        self.buffer = []
        self.count = 0
        
    def parse_audio(self, audio_path):
        return load_audio(audio_path)
        # if self.aug_conf and self.aug_conf.speed_volume_perturb:
        #     y = load_randomly_augmented_audio(audio_path, self.sample_rate)
        # else:
        #     y = load_audio(audio_path)
        # if self.noise_injector:
        #     add_noise = np.random.binomial(1, self.aug_conf.noise_prob)
        #     if add_noise:
        #         y = self.noise_injector.inject_noise(y)
        # n_fft = int(self.sample_rate * self.window_size)
        # win_length = n_fft
        # hop_length = int(self.sample_rate * self.window_stride)
        # # STFT
        # # print("y.shape: ", y.shape)
        # D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
        #                  win_length=win_length, window=self.window)
        # spect, phase = librosa.magphase(D)
        # # S = log(S+1)
        # spect = np.log1p(spect)
        # spect = torch.FloatTensor(spect)
        # if self.normalize:
        #     mean = spect.mean()
        #     std = spect.std()
        #     spect.add_(-mean)
        #     spect.div_(std)
        # # print("shape of spect: ", spect.shape)
        # if self.aug_conf and self.aug_conf.spec_augment:
        #     spect = spec_augment(spect)

        # return spect

    def __getitem__(self, index):
        if self.sample_length is None:
            sample = self.ids[index]
            audio_path, transcript_path = sample[0], sample[1]
            wav = self.parse_audio(audio_path)
            transcript = self.parse_transcript(transcript_path)
            return wav, transcript   
        if len(self.buffer) == 0:
            sample = self.ids[self.count]
            audio_path, transcript_path = sample[0], sample[1]
            wav = self.parse_audio(audio_path)
            transcript = self.parse_transcript(transcript_path)
            wav_len = wav.shape[0]
            repeat = wav_len // self.sample_length + 1
            pad = self.sample_length * repeat - wav_len
            wav = np.pad(wav,pad)
            for i in range(repeat):
                self.buffer.append(wav[i*self.sample_length: (i+1)*self.sample_length])
            self.count = (self.count + 1) // self.size
        wav = self.buffer.pop()
        return wav, "_"

    def _parse_input(self, input_path):
        ids = []
        if os.path.isdir(input_path):
            for wav_path in Path(input_path).rglob('*.wav'):
                transcript_path = str(wav_path).replace('/wav/', '/txt/').replace('.wav', '.txt')
                ids.append((wav_path, transcript_path))
        else:
            # Assume it is a manifest file
            with open(input_path) as f:
                manifest = json.load(f)
            for sample in manifest['samples']:
                wav_path = os.path.join(manifest['root_path'], sample['wav_path'])
                transcript_path = os.path.join(manifest['root_path'], sample['transcript_path'])
                ids.append((wav_path, transcript_path))
        return ids

    def parse_transcript(self, transcript_path):
        with open(transcript_path, 'r', encoding='utf8') as transcript_file:
            transcript = transcript_file.read().replace('\n', '')
        transcript = list(filter(None, [self.labels_map.get(x) for x in list(transcript)]))
        return transcript

    def __len__(self):
        return self.size


def _collate_fn(batch):
    def func(p):
        return p[0].size(1)

    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    longest_sample = max(batch, key=func)[0]
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    input_percentages = torch.FloatTensor(minibatch_size)
    target_sizes = torch.IntTensor(minibatch_size)
    targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(1)
        inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        input_percentages[x] = seq_length / float(max_seqlength)
        target_sizes[x] = len(target)
        targets.extend(target)
    targets = torch.tensor(targets, dtype=torch.long)
    return inputs, targets, input_percentages, target_sizes

class UniversalAttackDataModule(pl.LightningDataModule):
    
    def __init__(self,
                 labels: list,
                 data_cfg: DataConfig,
                 normalize: bool,
                 is_distributed: bool,
                 train_sample_length: int = None,
                 val_sample_length: int = None):
        super().__init__()
        self.train_path = to_absolute_path(data_cfg.train_path)
        self.val_path = to_absolute_path(data_cfg.val_path)
        self.labels = labels
        self.data_cfg = data_cfg
        # self.spect_cfg = data_cfg.spect
        # self.aug_cfg = data_cfg.augmentation
        self.normalize = normalize
        self.is_distributed = is_distributed
        self.train_sample_length = train_sample_length
        self.val_sample_length = val_sample_length

    def train_dataloader(self):
        train_dataset = self._create_dataset(self.train_path, self.train_sample_length)
        if self.is_distributed:
            train_sampler = DSElasticDistributedSampler(
                dataset=train_dataset,
                batch_size=self.data_cfg.batch_size
            )
        else:
            train_sampler = DSRandomSampler(
                dataset=train_dataset,
                batch_size=self.data_cfg.batch_size
            )
        train_loader = DataLoader(
            dataset=train_dataset,
            num_workers=self.data_cfg.num_workers,
            batch_sampler=train_sampler
        )
        return train_loader

    def val_dataloader(self):
        val_dataset = self._create_dataset(self.val_path,self.val_sample_length)
        val_loader = DataLoader(
            dataset=val_dataset,
            num_workers=self.data_cfg.num_workers,
            batch_size=1
        )
        return val_loader

    def _create_dataset(self, input_path, sample_length=None):
        dataset = WavDataset(
            input_path=input_path,
            labels=self.labels,
            normalize=True,
            sample_length=sample_length
            # aug_cfg=self.aug_cfg
        )
        return dataset
    
@dataclass
class DataConfig:
    train_path: str = 'data/train_manifest.csv'
    val_path: str = 'data/val_manifest.csv'
    batch_size: int = 64  # Batch size for training
    num_workers: int = 4  # Number of workers used in data-loading
    labels_path: str = 'labels.json'  # Contains tokens for model output