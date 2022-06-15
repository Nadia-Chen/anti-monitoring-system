# import imp
import sys, os

from numpy import double
sys.path.append("../deepspeech.pytorch/")
from deepspeech_pytorch.configs.inference_config import TranscribeConfig
from deepspeech_pytorch.utils import load_decoder, load_model
from attack import Attacker
import torchaudio
import argparse
import librosa
import torch
from deepspeech_pytorch.loader.data_loader import SpectrogramParser
from utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # I/O parameters
    parser.add_argument('--input_wav', type=str, help='input wav. file')
    parser.add_argument('--input_dir', type=str, help='input directory of wav. files')
    parser.add_argument('--output_wav', type=str, default='None', help='output adversarial wav. file')
    parser.add_argument('--model_path', type=str, default='/home/xuemeng/reverb/deepspeech.pytorch/models/librispeech/librispeech_pretrained_v3.ckpt', help='model pth path; please use absolute path')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    
    # attack parameters
    parser.add_argument('--target_sentence', type=str, default="HELLO WORLD", help='Please use uppercase')
    parser.add_argument('--mode', type=str, default="PGD", help='PGD or FGSM')
    parser.add_argument('--epsilon', type=float, default=0.001, help='epsilon')
    parser.add_argument('--alpha', type=float, default=1e-4, help='alpha')
    parser.add_argument('--num_iters', type=int, default=100, help='PGD iteration times')

    # plot parameters
    parser.add_argument('--plot_ori_spec', type=str, default="None", help='Path to save the original spectrogram')
    parser.add_argument('--plot_adv_spec', type=str, default="None", help='Path to save the adversarial spectrogram')
    args = parser.parse_args()

    cfg = TranscribeConfig
    model = load_model(device="cpu", model_path=args.model_path)
    decoder = load_decoder(labels=model.labels, cfg=cfg.lm)
    spect_parser = SpectrogramParser(
        audio_conf=model.spect_cfg,
        normalize=True
    )

    # sound, sample_rate = torchaudio.load(args.input_wav)
    sound, sample_rate = librosa.load(args.input_wav, sr=16000)
    sound = torch.tensor(sound)
    sound = torch.unsqueeze(sound, dim=0)
    print(sound.shape)
    target_sentence = args.target_sentence.upper()
    output_wav = args.output_wav
    if args.output_wav == "None":
        output_wav = None
        
    # train_loader = WavDataLoader(
    #     os.path.join(args.input_dir, "train"))
    # val_loader = WavDataLoader(
    #     os.path.join(args.input_dir, "valid"))    
    attacker = Attacker(model=model, sound=sound, target=target_sentence, decoder=decoder, spect_parser=spect_parser, device=args.device, save=output_wav)

    attacker.attack(epsilon = args.epsilon, alpha=args.alpha, attack_type=args.mode, PGD_round=args.num_iters)

    if args.plot_ori_spec != "None":
        attacker.get_ori_spec(args.plot_ori_spec)
    
    if args.plot_adv_spec != "None":
        attacker.get_adv_spec(args.plot_adv_spec )