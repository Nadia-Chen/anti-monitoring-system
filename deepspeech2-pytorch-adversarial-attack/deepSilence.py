import sys, os

from numpy import double
sys.path.append("../deepspeech.pytorch/")
from deepspeech_pytorch.configs.inference_config import TranscribeConfig
from deepspeech_pytorch.utils import load_decoder, load_model
from universal_attack import UniversalAttacker
import torchaudio
import argparse
import librosa
import torch
from deepspeech_pytorch.loader.data_loader import SpectrogramParser
from utils import *
from dataloader import UniversalAttackDataModule
from dataloader import DataConfig

from typing import List
from stft import STFT, magphase

from acoustic_feature import VAD

def target_sentence_to_label(sentence, labels="_'ABCDEFGHIJKLMNOPQRSTUVWXYZ "):
    out = []
    for word in sentence:
        out.append(labels.index(word))
    return torch.IntTensor(out)

def label_to_sentence(label, labels="_'ABCDEFGHIJKLMNOPQRSTUVWXYZ "):
    sentence = ""
    sentences = ""
    if isinstance(label, List):
        for i in range(len(label[0])):
            for word in label:
                word = word.numpy()
                word = word[i]
                sentence += labels[word]
    else:
        for i in range(label.shape[0]):
            for j in range(label.shape[1]):
                word = label[i][j]
                sentence += labels[word]
    return sentence

def gradients_status(model, flag):
    for p in model.parameters():
        p.requires_grad = flag

def torch_spectrogram(sound, torch_stft):
    # if sound.shape[0] == 1:
    #     sound = sound.squeeze()
    # else:
    #     sound = sound.mean(axis=0) 
    real, imag = torch_stft(sound)
    mag, cos, sin = magphase(real, imag)
    mag = torch.log1p(mag)
    mean = mag.mean()
    std = mag.std()
    mag = mag - mean
    mag = mag / std
    # print("mag shape before permute: ", mag.shape)
    mag = mag.permute(0,1,3,2)
    # print("mag shape after permute: ", mag.shape)
    return mag


def main():
    cfg = TranscribeConfig
    input_wav = "data/6078-54007-0037.wav"
    # input_wav = "/home/xuemeng/reverb/deepspeech2-pytorch-adversarial-attack/without_the_dataset_the_articel_is_useless.wav"
    perturbation = "/home/xuemeng/reverb/deepspeech2-pytorch-adversarial-attack/tar-_.min_loss-3.2033169269561768.loss-3.9514660835266113.eps-0.02.alp-0.001.wav"
    model_path = '/home/xuemeng/reverb/deepspeech.pytorch/models/librispeech/librispeech_pretrained_v3.ckpt'
    vad = True
    model = load_model(device="cpu", model_path=model_path)
    decoder = load_decoder(labels=model.labels, cfg=cfg.lm)
    spect_parser = SpectrogramParser(
        audio_conf=model.spect_cfg,
        normalize=True
    )
    device = torch.device("cuda" if cfg.model.cuda else "cpu")
    
    # sound, sample_rate = torchaudio.load(args.input_wav)
    # sound, sample_rate = librosa.load(args.input_wav, sr=16000)
    # sound = torch.tensor(sound)
    # sound = torch.unsqueeze(sound, dim=0)
    # print(sound.shape)
    # target_sentence = target_sentence.upper()
    output_wav = input_wav.split(".")[0] + "_{}disturbed.wav".format("vad_" if vad else "all_")
    output_wav_ori = input_wav.split(".")[0] + "_origin.wav"
    if "/" in output_wav:
        output_wav = output_wav.split("/")[-1]
    if output_wav == "None":
        output_wav = None
        
    # train_loader = WavDataLoader(
    #     os.path.join(args.input_dir, "train"))
    # val_loader = WavDataLoader(
    #     os.path.join(args.input_dir, "valid"))   
    
    data_cfg = DataConfig()
    data_cfg.batch_size = 1
    data_cfg.train_path = '/home/xuemeng/reverb/datasets/mini_librispeech/libri_test_clean_manifest.json'
    data_cfg.val_path = '/home/xuemeng/reverb/datasets/mini_librispeech/libri_val_manifest.json' 

    
    model = load_model(
        device=device,
        model_path=model_path
    )
    
    decoder = load_decoder(
        labels=model.labels,
        cfg=cfg.lm
    )
    audio_conf = model.spect_cfg
    sample_rate = audio_conf.sample_rate
    window_size = audio_conf.window_size
    window_stride = audio_conf.window_stride
    window = audio_conf.window.name
    n_fft = int(sample_rate * window_size)
    hop_length = int(sample_rate * window_stride)
    win_length = n_fft
    
    torch_stft =  STFT(n_fft=n_fft , hop_length=hop_length, win_length=win_length ,  window=window, center=True, pad_mode='reflect', freeze_parameters=True, device=device)
    
    short_pert, sr = torchaudio.load(perturbation)
    data, sr = torchaudio.load(input_wav)
    if data.shape[0]>1:
        data = data[0:1]
    data = data.to(device)
    short_pert_len = short_pert.shape[1]
    audio_len = data.shape[1]
    repeat = audio_len // short_pert_len
    l = [short_pert for _ in range(repeat)]
    # t = [short_target for _ in range(repeat)]
    pad = audio_len - repeat * short_pert_len
    # t_pad = int(short_target_len * pad // short_pert_len)
    l.append(short_pert[:, :pad])
    # t.append(short_target[:, :t_pad])
    pert = torch.cat(l, dim=1).to(device)
    # target = torch.cat(t, dim=1).to(self.device)
    e_low_multifactor=1.0
    zcr_multifactor=1.0
    if vad:
        print("perturb utterance only using vad")
        v = VAD(
            data[0].cpu().numpy(), sample_rate,
            min_interval=15,
            pt=False,
            e_low_multifactor=e_low_multifactor,
            zcr_multifactor=zcr_multifactor,
        )
        mask = v.mask()
        mask = torch.tensor(mask,dtype=torch.float32).to(device)
        pert = pert * mask
    ae = pert + data
    
    spec = torch_spectrogram(ae, torch_stft) # [1, 1, 161, 662] # ! [6, 1, 161, 101]
    # * input percentage
    input_percentages = torch.ones((repeat,)).to(device)
    input_sizes = torch.IntTensor([spec.size(3)]).int()
    # target_sizes = torch.ones((repeat,)).to(device)
    # target_lengths = target_sizes.mul_(target.shape[1]).int()
    # input_sizes = torch.IntTensor([spec.size(3)]).int() # [662] # ! [101]
    # self.model.train()
    out, output_sizes = model(spec, input_sizes) # [1, 331, 29], [331]
    decoded_output, decoded_offsets = decoder.decode(out, output_sizes)
    output = decoded_output[0][0]
    print(f"Perturbed transcription: \'{output}\'")
    
    torchaudio.save(output_wav, src=ae.cpu(), sample_rate=sample_rate)
    torchaudio.save(output_wav_ori, src=data.cpu(), sample_rate=sample_rate)
    
      


if __name__ == "__main__":
    main()