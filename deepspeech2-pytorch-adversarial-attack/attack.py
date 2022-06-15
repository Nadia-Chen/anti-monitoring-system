from stft import STFT, magphase
import torch.nn as nn
import torch
import Levenshtein
import torchaudio
import numpy as np
import matplotlib.pyplot as plt

from deepspeech_pytorch.configs.inference_config import TranscribeConfig
from typing import List
from torch.cuda.amp import autocast

def target_sentence_to_label(sentence, labels="_'ABCDEFGHIJKLMNOPQRSTUVWXYZ "):
    out = []
    for word in sentence:
        out.append(labels.index(word))
    return torch.IntTensor(out)

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

# # @hydra.main(config_path='.', config_name="config")
# def decode_results(decoded_output: List,
#                    decoded_offsets: List,
#                    cfg: TranscribeConfig):
#     results = {
#         "output": [],
#         "_meta": {
#             "acoustic_model": {
#                 "path": cfg.model.model_path
#             },
#             "language_model": {
#                 "path": cfg.lm.lm_path
#             },
#             "decoder": {
#                 "alpha": cfg.lm.alpha,
#                 "beta": cfg.lm.beta,
#                 "type": cfg.lm.decoder_type.value,
#             }
#         }
#     }

#     for b in range(len(decoded_output)):
#         for pi in range(min(cfg.lm.top_paths, len(decoded_output[b]))):
#             result = {'transcription': decoded_output[b][pi]}
#             if cfg.offsets:
#                 result['offsets'] = decoded_offsets[b][pi].tolist()
#             results['output'].append(result)
#     return results


class Attacker:
    def __init__(self, model, sound, target, decoder, spect_parser,sample_rate=16000, device="cpu", save=None):
        """
        model: deepspeech model
        sound: raw sound data [-1 to +1] (read from torchaudio.load)
        label: string
        """
        self.sound = sound
        self.sample_rate = sample_rate
        self.target_string = target
        self.target = target
        # self.save = "{}-2-{}-eps{}-alf{}.wav".format()
        self.init_target()
        
        # * extract audio config from checkpoint model
        self.audio_conf = model.spect_cfg
        self.sample_rate = self.audio_conf.sample_rate
        self.window_size = self.audio_conf.window_size
        self.window_stride = self.audio_conf.window_stride
        self.window = self.audio_conf.window.name
        
        self.model = model
        self.model.to(device)
        # self.model.train()
        self.model.train()
        # gradients_status(self.model, False)
        # self.model.rnns.training = True
        # self.model.rnns.train()
        for idx, rnn in enumerate(self.model.rnns):
            if idx == 0:
                continue
            rnn.batch_norm.eval()
            
        for module in self.model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
        self.decoder = decoder
        self.spect_parser = spect_parser
        self.criterion = nn.CTCLoss()
        self.device = device
        n_fft = int(self.sample_rate * self.window_size)
        hop_length = int(self.sample_rate * self.window_stride)
        win_length = n_fft
        self.torch_stft = STFT(n_fft=n_fft , hop_length=hop_length, win_length=win_length ,  window=self.window, center=True, pad_mode='reflect', freeze_parameters=True, device=self.device)
        self.save = save
        
        
    def get_ori_spec(self, save=None):
        spec = torch_spectrogram(self.sound.to(self.device), self.torch_stft)
        plt.imshow(spec.cpu().numpy()[0][0])
        if save:
            plt.savefig(save)
            plt.clf()
        else:
            plt.show()

    def get_adv_spec(self, save=None):
        spec = torch_spectrogram(self.perturbed_data.to(self.device), self.torch_stft)
        plt.imshow(spec.cpu().numpy()[0][0])
        if save:
            plt.savefig(save)
            plt.clf()
        else:
            plt.show()
    
    # prepare
    def init_target(self):
        self.target = target_sentence_to_label(self.target)
        self.target = self.target.view(1,-1)
        self.target_lengths = torch.IntTensor([self.target.shape[1]]).view(1,-1)

    # FGSM
    def fgsm_attack(self, sound, epsilon, data_grad):
        
        # find direction of gradient
        sign_data_grad = data_grad.sign()
        
        # add noise "epilon * direction" to the ori sound
        perturbed_sound = sound - epsilon * sign_data_grad
        
        return perturbed_sound
    
    # PGD
    def pgd_attack(self, sound, ori_sound, eps, alpha, data_grad) :
        
        adv_sound = sound - alpha * data_grad.sign() # + -> - !!!
        eta = torch.clamp(adv_sound - ori_sound.data, min=-eps, max=eps)
        sound = ori_sound + eta

        return sound
    
    def attack(self, epsilon, alpha, attack_type = "FGSM", PGD_round=40):        

        data, target = self.sound.to(self.device), self.target.to(self.device)
        data_raw = data.clone().detach()
        
        # initial prediction
        spec = torch_spectrogram(data, self.torch_stft)
        # self.spect_parser.parse_audio()
        input_sizes = torch.IntTensor([spec.size(3)]).int()
        # precision = 32 # !
        # with autocast(enabled=precision == 16):
        out, output_sizes = self.model(spec, input_sizes) # [1, 266, 29]
        decoded_output, decoded_offsets = self.decoder.decode(out, output_sizes)
        original_output = decoded_output[0][0]
        print(f"Original prediction: {original_output}")
        # decode_results(
        #     decoded_output=decoded_output,
        #     decoded_offsets=decoded_offsets
        # )
        
        # ATTACK
        ############ ATTACK GENERATION ##############
        if attack_type == "FGSM":
            data.requires_grad = True
            
            spec = torch_spectrogram(data, self.torch_stft)
            input_sizes = torch.IntTensor([spec.size(3)]).int()
            out, output_sizes = self.model(spec, input_sizes)
            out = out.transpose(0, 1)  # TxNxH
            out = out.log_softmax(2)
            loss = self.criterion(out, self.target, output_sizes, self.target_lengths)
            
            self.model.zero_grad()
            loss.backward()
            data_grad = data.grad.data

            perturbed_data = self.fgsm_attack(data, epsilon, data_grad)

        elif attack_type == "PGD":
            for i in range(PGD_round):
                print(f"PGD processing ...  {i+1} / {PGD_round}", end="\r")
                data.requires_grad = True
                
                spec = torch_spectrogram(data, self.torch_stft)
                input_sizes = torch.IntTensor([spec.size(3)]).int()
                # self.model.train()
                out, output_sizes = self.model(spec, input_sizes)
                out = out.transpose(0, 1)  # TxNxH
                out = out.log_softmax(2)
                loss = self.criterion(out, self.target, output_sizes, self.target_lengths) # out: [266, 1, 29], target: [1, 13], output_size: 266, target_length: 13
                # self.model.train()
                # self.model.zero_grad()
                print(f"loss {i}: {loss}")
                loss.backward()
                data_grad = data.grad.data

                data = self.pgd_attack(data, data_raw, epsilon, alpha, data_grad).detach_()
            perturbed_data = data
        ############ ATTACK GENERATION ##############

        # prediction of adversarial sound
        spec = torch_spectrogram(perturbed_data, self.torch_stft)
        input_sizes = torch.IntTensor([spec.size(3)]).int()
        # self.model.eval()
        out, output_sizes = self.model(spec, input_sizes)
        decoded_output, decoded_offsets = self.decoder.decode(out, output_sizes)
        final_output = decoded_output[0][0]
        # print(f"Final prediction: {original_output}")
        
        perturbed_data = perturbed_data.detach()
        abs_ori = 20*np.log10(np.sqrt(np.mean(np.absolute(data_raw.cpu().numpy())**2)))
        abs_after = 20*np.log10(np.sqrt(np.mean(np.absolute(perturbed_data.cpu().numpy())**2)))
        db_difference = abs_after-abs_ori
        l_distance = Levenshtein.distance(self.target_string, final_output)
        print(f"Max Decibel Difference: {db_difference:.4f}")
        print(f"Adversarial prediction: {decoded_output[0][0]}")
        print(f"Levenshtein Distance {l_distance}")
        if len(original_output) > 20:
            original_output = original_output[:20]+ "..."
        if len(self.target_string) > 20:
            self.target_string = self.target_string[:20]+ "..."
        if len(final_output) > 20:
            final_output = final_output[:20]+ "..."
        self.save = "ori-{}.tar-{}.fin-{}-eps{}-alp{}.wav".format(original_output.replace(" ", "_"), self.target_string.replace(" ", "_"), final_output.replace(" ", "_"), epsilon, alpha)
        if self.save:
            torchaudio.save(self.save, src=perturbed_data.cpu(), sample_rate=self.sample_rate)
        self.perturbed_data = perturbed_data
        return db_difference, l_distance, self.target_string, final_output