from statistics import mean
from stft import STFT, magphase
import torch.nn as nn
import torch
import Levenshtein
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import os

from deepspeech_pytorch.configs.inference_config import TranscribeConfig
from typing import List
from torch.cuda.amp import autocast

from dataloader import UniversalAttackDataModule
import torch.optim as optim

from acoustic_feature import VAD
from unet_basic import UNet

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


class UNetAttacker:
    def __init__(self, model, data_loader,target, decoder, spect_parser,sample_rate=16000, device="cpu", save=None, resume=None):
        """
        model: deepspeech model
        sound: raw sound data [-1 to +1] (read from torchaudio.load)
        label: string
        """
        self.data_loader = data_loader
        self.train_loader = iter(data_loader.train_dataloader())
        self.val_loader = iter(data_loader.val_dataloader())
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
        # * set batch norm layers to evaluate mode
        for idx, rnn in enumerate(self.model.rnns):
            if idx == 0:
                continue
            rnn.batch_norm.eval()
            
        for module in self.model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
        self.decoder = decoder
        self.spect_parser = spect_parser
        
        
        self.beta = 0.1
        self.ctc_loss = nn.CTCLoss()
        self.l2_loss = nn.MSELoss()
        
        
        self.device = device
        n_fft = int(self.sample_rate * self.window_size)
        hop_length = int(self.sample_rate * self.window_stride)
        win_length = n_fft
        self.torch_stft = STFT(n_fft=n_fft , hop_length=hop_length, win_length=win_length ,  window=self.window, center=True, pad_mode='reflect', freeze_parameters=True, device=self.device)
        self.save = save
        self.resume = resume
        
        
        self.e_low_multifactor = 1.0
        self.zcr_multifactor = 1.0
        
        # * set up U-Net model for the attack
        self.unet = UNet()
        self.unet.to(self.device)
        self.unet_optimizer = torch.optim.Adam(
            params=self.unet.parameters(),
            lr=1e-4,
            # betas=(0.1, 0.1)
        )
        # TODO: learning rate decay
        # self.lr_schedular = torch.optim.lr_scheduler.
        
        # * valid 
        self.valid_interval = 100
        self.checkpoint_interval = 100
        
        
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
    def pgd_attack(self, short_pert, eps, alpha, pert_grad) :
        
        short_pert = short_pert - alpha * pert_grad.sign() # + -> - !!!
        short_pert = torch.clamp(short_pert, min=-eps, max=eps)
        # pert = short_pert + eta

        return short_pert
    
    def attack(self, epsilon, alpha, attack_type = "FGSM", num_iteration=40, perturbation_len=1.0, batch=None, optimization=None, vad=False):

        short_target = self.target.to(self.device)
        short_target_len = short_target.shape[1]
        # data_raw = data.clone().detach()
        short_pert_len = int(perturbation_len * 16384 * 2)
        # short_pert_len = int(perturbation_len * self.sample_rate)
             
        feature_map = None
        if self.resume is not None:
            raise Exception("resume from check point not implemented")
        else:
            feature_map = torch.zeros((1, short_pert_len)).to(self.device)
            print(f"Start training Wave-U-Net from the start")
            feature_map = torch.FloatTensor(1, short_pert_len).to(self.device)
            feature_map.data.normal_(mean=0, std=epsilon)
        feature_map.requires_grad = True

        # * create optimizer
        optimizer = None
        if optimization == "adam":
            feature_map = nn.Parameter(feature_map,requires_grad=True)
            print(f"optimize with \'{optimization}\'")
            optimizer = optim.Adam([feature_map])
        
        # initial prediction
        # with torch.no_grad():
        spec = torch_spectrogram(feature_map, self.torch_stft)
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
        min_loss = 10000
        latest_loss = 0
        # Start training
        for i in range(num_iteration):
            print(f"Wave-U-Net Training ...  {i+1} / {num_iteration}", end="\r")
            self.unet.train()
            # * load data
            try:
                data, label = next(self.train_loader)
            except Exception:
                self.train_loader = iter(self.data_loader.train_dataloader())
                data, label = next(self.train_loader)
            sentence = "_"
            data = data.to(self.device)
            data_raw = data.clone().detach()
            # data.requires_grad = True
            
            # * construct perturbation according to length
            audio_len = data.shape[1]
            pert = None

            # * restrict magnitude of the short perturbation
            short_pert_clamp = torch.clamp(feature_map, min=-epsilon, max=epsilon)

            if audio_len < short_pert_len:
                pert = short_pert_clamp[:, :audio_len]
            else:
                repeat = audio_len // short_pert_len
                if not batch: # contract batch [audio_len // short_pert_len: short_pert_len]
                    # repeat = 1
                    l = [short_pert_clamp for _ in range(repeat)]
                    t = [short_target for _ in range(repeat)]
                    pad = audio_len - repeat * short_pert_len
                    t_pad = int(short_target_len * pad // short_pert_len)
                    l.append(short_pert_clamp[:, :pad])
                    t.append(short_target[:, :t_pad])
                    pert = torch.cat(l, dim=1).to(self.device)
                    target = torch.cat(t, dim=1).to(self.device)
                    if vad:
                        v = VAD(
                            data[0].cpu().numpy(), self.sample_rate,
                            min_interval=15,
                            pt=False,
                            e_low_multifactor=self.e_low_multifactor,
                            zcr_multifactor=self.zcr_multifactor,
                        )
                        mask = v.mask()
                        mask = torch.tensor(mask,dtype=torch.float32).to(self.device)
                        pert = pert * mask
                    # print(mask.sum())
                    
                else:
                    batch = data.shape[0]
                    # stack data and target into 2 dimensional 
                    # d = [data[:, i * short_pert_len: (i+1) * short_pert_len] for i in range(repeat)]
                    # d.append(data[:, repeat * short_pert_len])
                    data = torch.unsqueeze(data,dim=1)
                    t = [short_target[0] for _ in range(batch)]
                    # data = torch.stack(d)
                    
                    # TODO: embed unet with vector
                    
                    # * audio into unet
                    # self.unet.eval()
                    with torch.no_grad():
                        pertubed = self.unet(data)
                    clamped_perturb = torch.clamp(pertubed, min=-epsilon, max=epsilon)
                    # ae = clamped_perturb + data
                    target = torch.stack(t)
                    pert = clamped_perturb
            ae = pert + data # [1, 105760]
            ae = ae[:, 0, :]
            pert.retain_grad()
            spec = torch_spectrogram(ae, self.torch_stft) # [1, 1, 161, 662] # ! [6, 1, 161, 101]
            # * input percentage
            
            if batch:
                input_percentages = torch.ones((batch,)).to(self.device)
                input_sizes = input_percentages.mul_(int(spec.size(3))).int()
            else:
                input_sizes = torch.IntTensor([spec.size(3)]).int() 
            target_sizes = torch.ones((batch if batch else 1,)).to(self.device)
            target_lengths = None
            if batch:
                target_lengths = target_sizes.mul_(self.target.shape[1]).int()
            else:
                target_lengths = target_sizes.mul_(target.shape[1]).int()
            # input_sizes = torch.IntTensor([spec.size(3)]).int() # [662] # ! [101]
            # self.model.train()
            out, output_sizes = self.model(spec, input_sizes) # [1, 331, 29], [331]
            decoded_output, decoded_offsets = self.decoder.decode(out, output_sizes)
            output = decoded_output[0][0]
            out = out.transpose(0, 1)  # TxNxH
            out = out.log_softmax(2)
            zeros = torch.zeros_like(pert)
            ctc_loss = self.ctc_loss(out, target, output_sizes,  target_lengths)
            l2_loss = self.l2_loss(pert, zeros) # out: [266, 1, 29], target: [1, 13], output_size: 266, target_length: 13
            loss = ctc_loss + self.beta * l2_loss
            # self.model.train()
            self.model.zero_grad()
            # original_sentence = label_to_sentence(target)
            if loss < min_loss:
                min_loss = loss
            latest_loss = loss
            print(f"itr {i}: ctc loss: {ctc_loss}, l2_loss: {l2_loss}")
            # print(f"\toriginal     : \'{sentence}\'")
            # print(f"\ttarget       : \'{self.target_string}\'")
            # print(f"\ttranscription: \'{output}\'")
            # * save successful samples
            # if self.target_string == output:
            #     if len(sentence) > 20:
            #         sentence = sentence[0:20] + "..."
            #     file_name = "ori-{}.tras-{}.wav".format(sentence.replace(" ", "_"), output.replace(" ", "_"))
            #     path = os.path.join("aes", file_name)
            #     torchaudio.save(path, src=ae[0:1].cpu(), sample_rate=self.sample_rate)
                
            loss.backward()
            if optimization is None:
                pert_grad = feature_map.grad.data

                feature_map = self.pgd_attack(feature_map, epsilon, alpha, pert_grad).detach_()
                feature_map.requires_grad = True
            elif optimization == "adam":
                optimizer.step()
                optimizer.zero_grad()
                # short_pert.zero_grad()
            else:
                raise Exception("Optimizer not implemented")
            self.unet_optimizer.step()
            self.unet_optimizer.zero_grad()
            
            # TODO: valid
            if i % self.valid_interval == 0:
                self.unet.eval()
                # try:
                data, label = next(self.val_loader)
                # except Exception:
                #     self.val_loader = iter(self.data_loader.val_dataloader())
                #     data, label = next(self.val_loader)
                data = data.to(self.device)
                sentence = label_to_sentence(label)
                
                audio_len = data.shape[1]
                pert = None
                repeat = audio_len // short_pert_len
                if audio_len < short_pert_len:
                    pert = short_pert_clamp[:, :audio_len]
                else:
                    
                    # repeat = 1
                    # l = [short_pert_clamp for _ in range(repeat)]
                    d = [data[0, i * short_pert_len: (i+1) * short_pert_len] for i in range(repeat)]
                    data = torch.stack(d)
                    data = torch.unsqueeze(data,dim=1)
                    t = [short_target for _ in range(repeat)]
                    pad = audio_len - repeat * short_pert_len
                    t_pad = int(short_target_len * pad // short_pert_len)
                    # l.append(short_pert_clamp[:, :pad])
                    t.append(short_target[:, :t_pad])
                    # data = data[:, :, :16384]
                    perturbed = self.unet(data)
                    pert = torch.clamp(perturbed, min=-epsilon, max=epsilon)
                    # pert = torch.cat(l, dim=1).to(self.device)
                    target = torch.cat(t, dim=1).to(self.device)
                    # target = torch.stack(t)
                    # if vad:
                    #     v = VAD(
                    #         data[0].cpu().numpy(), self.sample_rate,
                    #         min_interval=15,
                    #         pt=False,
                    #         e_low_multifactor=self.e_low_multifactor,
                    #         zcr_multifactor=self.zcr_multifactor,
                    #     )
                    #     mask = v.mask()
                    #     mask = torch.tensor(mask,dtype=torch.float32).to(self.device)
                    #     pert = pert * mask
                    # print(mask.sum())

                ae = pert + data # [1, 105760]
                # ae = ae[:, 0, :]
                ae = torch.reshape(ae, (1,-1))
                # ae = torch.flatten(ae, 0)
                # pert.retain_grad()
                spec = torch_spectrogram(ae, self.torch_stft) # [1, 1, 161, 662] # ! [6, 1, 161, 101]
                # * input percentage
                # input_percentages = torch.ones((repeat,)).to(self.device)
                # input_sizes = input_percentages.mul_(int(spec.size(3))).int()
                input_sizes = torch.IntTensor([spec.size(3)]).int() 
                target_sizes = torch.ones((1,)).to(self.device)
                target_lengths = None
                target_lengths = target_sizes.mul_(target.shape[1]).int()
                # input_sizes = torch.IntTensor([spec.size(3)]).int() # [662] # ! [101]
                # self.model.train()
                out, output_sizes = self.model(spec, input_sizes) # [1, 331, 29], [331]
                decoded_output, decoded_offsets = self.decoder.decode(out, output_sizes)
                # output = decoded_output[0][0]
                output = ''
                for string in decoded_output:
                    output += string[0]
                out = out.transpose(0, 1)  # TxNxH
                out = out.log_softmax(2)
                zeros = torch.zeros_like(pert)
                ctc_loss = self.ctc_loss(out, target, output_sizes,  target_lengths)
                l2_loss = self.l2_loss(pert, zeros) # out: [266, 1, 29], target: [1, 13], output_size: 266, target_length: 13
                loss = ctc_loss + self.beta * l2_loss
                # self.model.train()
                self.model.zero_grad()
                # original_sentence = label_to_sentence(target)
                if loss < min_loss:
                    min_loss = loss
                latest_loss = loss
                print(f"itr {i}: ctc loss: {ctc_loss}, l2_loss: {l2_loss}")
                print(f"\toriginal     : \'{sentence}\'")
                print(f"\ttarget       : \'{self.target_string}\'")
                print(f"\ttranscription: \'{output}\'")
                    
                    
            
            
            # TODO: decrease epsilon
                
            # perturbed_data = data
        ############ ATTACK GENERATION ##############

        # prediction of adversarial sound
        # spec = torch_spectrogram(perturbed_data, self.torch_stft)
        # input_sizes = torch.IntTensor([spec.size(3)]).int()
        # # self.model.eval()
        # out, output_sizes = self.model(spec, input_sizes)
        # decoded_output, decoded_offsets = self.decoder.decode(out, output_sizes)
        # final_output = decoded_output[0][0]
        spec = torch_spectrogram(feature_map, self.torch_stft)
        # self.spect_parser.parse_audio()
        input_sizes = torch.IntTensor([spec.size(3)]).int()
        # precision = 32 # !
        # with autocast(enabled=precision == 16):
        out, output_sizes = self.model(spec, input_sizes) # [1, 266, 29]
        decoded_output, decoded_offsets = self.decoder.decode(out, output_sizes)
        final_output = decoded_output[0][0]
        print(f"Final prediction: {final_output}")
        # print(f"Final prediction: {original_output}")
        
        # perturbed_data = perturbed_data.detach()
        # abs_ori = 20*np.log10(np.sqrt(np.mean(np.absolute(data_raw.cpu().numpy())**2)))
        # abs_after = 20*np.log10(np.sqrt(np.mean(np.absolute(perturbed_data.cpu().numpy())**2)))
        # db_difference = abs_after-abs_ori
        l_distance = Levenshtein.distance(self.target_string, final_output)
        # print(f"Max Decibel Difference: {db_difference:.4f}")
        print(f"Adversarial prediction: {decoded_output[0][0]}")
        print(f"Levenshtein Distance {l_distance}")
        self.save = "tar-{}.min_loss-{}.loss-{}.vad-{}.eps-{}.alp-{}.{}.optim-{}.wav".format(self.target_string.replace(" ", "_"), min_loss, latest_loss, "True" if vad else "false", epsilon, alpha, "batch" if batch else "nbatch", attack_type if optimization is None else optimization)
        if self.save:
            torchaudio.save(self.save, src=feature_map.cpu(), sample_rate=self.sample_rate)
        self.perturbed_data = feature_map
        return 0.0, l_distance, self.target_string, final_output