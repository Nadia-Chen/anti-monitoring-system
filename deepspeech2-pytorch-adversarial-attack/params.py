import torch
import random
import numpy as np
import logging
import os
import shutil

#! 原语音分类以及目标分类
target = 2  # Target command id
# label = CLASSES[target]  # Path of the target labels
original = 2  # original command id
CLASSES = "unknown, silence, yes, no, up, down, left, right, on, off, stop, go".split(
    ", "
)
early_stop = False
train_mode = "one_origin"  # ! "one_task", "one_origin", "one_target", "all_tasks"
#############################
# DataSet Path
#
# ! rever dataset
target_signals_dir = "./data/{}_0.2".format(CLASSES[original])
rirs_noises = True
# * data set for physical attack
#rir_dir = "/project/xuemeng/reverb/echo/input_0.2/rir"
rir_dir = ""
if not rirs_noises:
    rir_dir = "/home/xuemeng/reverb/echo/yes_0.2/rir"
else:
    rir_dir = "/home/xuemeng/Proj/SRDebug/data/RIRS_NOISES/simulated_rirs"
# rir_dir = "data/rir/"
# rir_dir = "/home/xuemeng/Proj/SRDebug/data/RIRS_NOISES/simulated_rirs/largeroom"
noise_dir = "/home/xuemeng/Proj/SRDebug/data/RIRS_NOISES/pointsource_noises"


#############################
# * Adversarial Params
#############################

# Original audio for generating adversarial example
input = "{}.wav".format(CLASSES[original])
target_model_name = "vgg19_bn"
target_model = "best-acc-speech-commands-checkpoint-vgg19_bn_sgd_plateau_bs96_lr1.0e-02_wd1.0e-02.pth"


# ================
reverb_bias = 0.1  # ! delay of reverb
adv_magnitude = 0.1  # Parameter specifing how much the perturbation be diminished
# rt60 = 0  # ! How long the reverb last
confidence = 0.01  # Parameter specifing how strong the adversarial example should be
adv_loss_lambda = 0.5  # Lambda for adversarial loss
z_mask = False  # ! whether use restricted mask
mask_length = 0.5  # ! restricted mask length
pretrained = False  # ! use pretrained model or not

low_pass = False  # ! low pass
low_pass_cutoff = 8000

physical = True  # ! physical attack or digital attack

# ! vad params
e_low_multifactor = 1.0
zcr_multifactor = 1.0

# if not os.path.exists(target_signals_dir):
#     raise Exception("Reverb Dataset Not Found.")
# * nes configurations
adv_loss_only = False  # ! whether only use adv_loss or not
use_nes = True  # ! white box or black box
algorithm = "nes_s"
num_queries = 100
num_itrs = 16
i_fsgm = True
relative_grad = False
epsilon = 0.5
sigma = 0.6
#sigma = 0.001
# epsilon = 5e-07

# * physical settings
noise_ratio = 1.0


# *  default params of feature extraction of speech command model
n_mels = 32
input_length = 16000  # input audio length of target model

#############################
# Model Params
#############################

# model_prefix = "exp_"  # name of the model to be saved
n_iterations = 100000
use_batchnorm = False
lr_factor = 1.0  # * scalse learning rate factor
lr_g = 1e-4 * lr_factor
lr_d = (
    3e-4 * lr_factor
)  # you can use with discriminator having a larger learning rate than generator instead of using n_critic updates ttur https://arxiv.org/abs/1706.08500
beta1 = 0.5
beta2 = 0.9
decay_lr = (
    False  # used to linearly deay learning rate untill reaching 0 at iteration 100,000
)
# * learing rate strategy 'CyclicLR'
lr_schedule = "None"
# in some cases we might try to update the generator with double batch size used in the discriminator https://arxiv.org/abs/1706.08500
generator_batch_size_factor = 1
n_critic = 1  # update generator every n_critic steps if lr_g = lr_d the n_critic's default value is 5
# gradient penalty regularization factor.
validate = False
p_coeff = 10
batch_size = 64  # !batch size
noise_latent_dim = 100  # size of the sampling noise
# model capacity during training can be reduced to 32 for larger window length of 2 seconds and 4 seconds
model_capacity_size = 32
# rate of storing validation and costs params
store_cost_every = 300
progress_bar_step_iter_size = 400

num_save_samples = 63

# # * {model_type}_{adv_loss_lambda}_{batch_size}_{input_wave}_{use_batchnorm}_{target_idx}_{model_capacity_size}
# model_prefix = "P-{}-{}-{}-{}-{}-{}-{}-{}-medium".format(target_model_name, adv_loss_lambda, batch_size, input.split(
#     "/")[-1].split(".")[0], target, model_capacity_size, lr_factor, "nes_{}_{}_{}".format("ifsgm" if i_fsgm else "fsgm", "rel" if relative_grad else "abs", num_queries) if use_nes else "white")  # name of the model to be saved, loss going down model

# model_prefix = "{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-rt_{}".format("bp" if pretrained else "p", target_model_name, adv_loss_lambda, batch_size, input.split(
#     "/")[-1].split(".")[0], target, model_capacity_size, lr_factor, "nes_{}_{}_{}_{}".format("ifsgm" if i_fsgm else "fsgm", "rel" if relative_grad else "abs", num_queries, epsilon) if use_nes else "white", "zmask_{}".format(mask_length) if z_mask else "nmask", rt60)  # name of the model to be saved, new version

#! lowpass method
# model_prefix = "{}-{}-{}-{}-{}-{}-{}-{}-{}-{}".format("bp" if pretrained else "p", target_model_name, adv_loss_lambda, batch_size, input.split(
#     "/")[-1].split(".")[0], target, model_capacity_size, lr_factor, "nes_{}_{}_{}_{}".format("ifsgm" if i_fsgm else "fsgm", "rel" if relative_grad else "abs", num_queries, epsilon) if use_nes else "white", "vad_{}_{}".format(e_low_multifactor, zcr_multifactor))  # name of the model to be saved, new version

# * name of model to be saved: {use pretrained model or not}-{adv_loss_lambda}-{batch_size}-{input audio}-{target label}-{model_capacity_size}-{lr_factor}-{black or white box attack and configurations}-{vad params}-{use lowpass filter or not}
model_prefix = "final-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}{}{}".format(
    algorithm + ("_p" if physical else "_d"),
    target_model_name,
    adv_loss_lambda,
    batch_size,
    input.split("/")[-1].split(".")[0],
    "o{}_t{}".format(original, target),
    lr_factor,
    "evo_{}_{}_{}_{}_{}_{}".format(
        "ifsgm" if i_fsgm else "fsgm",
        "rel" if relative_grad else "abs",
        num_queries,
        num_itrs,
        epsilon,
        sigma
    )
    if use_nes
    else "white",
    "vad_{}_{}".format(e_low_multifactor, zcr_multifactor),
    "low_pass" if low_pass else "n_low_pass",
    "-zmask_{}".format(mask_length) if z_mask else "", "-adv_only" if adv_loss_only else ""
)  # name of the model to be saved, new version
# TODO 从evo开始又回到了只在进化算法开头乘上sigma

#! z_mask method
# model_prefix = "{}-{}-{}-{}-{}-{}-{}-{}-{}-{}".format("bp" if pretrained else "p", target_model_name, adv_loss_lambda, batch_size, input.split(
#     "/")[-1].split(".")[0], target, model_capacity_size, lr_factor, "nes_{}_{}_{}_{}".format("ifsgm" if i_fsgm else "fsgm", "rel" if relative_grad else "abs", num_queries, epsilon) if use_nes else "white", "zmask_{}".format(mask_length) if z_mask else "nmask")  # name of the model to be saved, new version

# train on previously working model
# model_prefix = "bp-vgg19_bn-0.05-64-input-10-32-1.0-nes_ifsgm_abs_100"
# model_prefix = "bp1-vgg19_bn-0.05-64-input-10-32-1.0-nes_ifsgm_abs_100"
# model_prefix = "bp-vgg19_bn-0.01-64-input-10-32-1.0-nes_ifsgm_abs_100_0.005"
# model_prefix = "bp-vgg19_bn-0.01-64-input-10-32-1.0-nes_ifsgm_abs_100_0.005_zmask"

# model_prefix = "E-{}-{}-{}-{}-{}-{}-{}-{}".format(target_model, adv_loss_lambda, batch_size, input.split("/")[-1].split(
#     ".")[0], target, model_capacity_size, lr_factor, "nes" if use_nes else "white")  # name of the model to be saved
#############################
# Backup Params
#############################
take_backup = True
backup_every_n_iters = 300
save_samples_every = 1000
output_dir = "./generated_clean_asr/output-" + model_prefix
trigger_dir = "./generated_clean_asr/tirgger-" + model_prefix
ae_dir = "./generated_clean_asr/ae-" + model_prefix
physical_dir = "./generated_clean_asr/phy-" + model_prefix
tbx_dir = "./result_clean_asr/" + model_prefix
gan_dir = "./gan_model/"
# copy pretrained model
# if pretrained and not os.path.isfile(
#     os.path.join(gan_dir, "gan_{}.tar".format(model_prefix))
# ):
#     shutil.copyfile(
#         "gan_model/gan_A-vgg19_bn-0.05-64-input-10-32-1.0-nes_ifsgm_abs_100.tar",
#         os.path.join(gan_dir, "gan_{}.tar".format(model_prefix)),
#     )
#if not (os.path.isdir(output_dir)):
#    os.makedirs(output_dir)
#if not (os.path.isdir(trigger_dir)):
#    os.makedirs(trigger_dir)
#if not (os.path.isdir(ae_dir)):
#    os.makedirs(ae_dir)
#if not (os.path.isdir(physical_dir)):
#    os.makedirs(physical_dir)
#if not (os.path.isdir(tbx_dir)):
#    os.makedirs(tbx_dir)


# def initialize(o, t):
#     # global variables
#     global original
#     global target
#     global target_signals_dir
#     global input
#     global model_prefix
#     global output_dir
#     global trigger_dir
#     global ae_dir
#     global physical_dir
#     global tbx_dir
#     global gan_dir

#     original = o
#     target = t

#     target_signals_dir = "./data/{}_0.2".format(CLASSES[original])
#     if not os.path.exists(target_signals_dir):
#         raise Exception("Reverb Dataset Not Found.")
#     input = "{}.wav".format(CLASSES[original])

#     model_prefix = "final-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}{}{}".format(
#         algorithm + ("_p" if physical else "_d"),
#         target_model_name,
#         adv_loss_lambda,
#         batch_size,
#         input.split("/")[-1].split(".")[0],
#         "o{}_t{}".format(original, target),
#         lr_factor,
#         "evo_{}_{}_{}_{}_{}_{}".format(
#             "ifsgm" if i_fsgm else "fsgm",
#             "rel" if relative_grad else "abs",
#             num_queries,
#             num_itrs,
#             epsilon,
#             sigma
#         )
#         if use_nes
#         else "white",
#         "vad_{}_{}".format(e_low_multifactor, zcr_multifactor),
#         "low_pass" if low_pass else "n_low_pass",
#         "-zmask_{}".format(mask_length) if z_mask else "", "-adv_only" if adv_loss_only else ""
#     )  # name of the model to be saved, new version

#     output_dir = "./generated_clean_asr/output-" + model_prefix
#     trigger_dir = "./generated_clean_asr/tirgger-" + model_prefix
#     ae_dir = "./generated_clean_asr/ae-" + model_prefix
#     physical_dir = "./generated_clean_asr/phy-" + model_prefix
#     tbx_dir = "./result_clean_asr/" + model_prefix
#     gan_dir = "./gan_model/"
#     # copy pretrained model
#     # if pretrained and not os.path.isfile(
#     #     os.path.join(gan_dir, "gan_{}.tar".format(model_prefix))
#     # ):
#     #     shutil.copyfile(
#     #         "gan_model/gan_A-vgg19_bn-0.05-64-input-10-32-1.0-nes_ifsgm_abs_100.tar",
#     #         os.path.join(gan_dir, "gan_{}.tar".format(model_prefix)),
#     #     )
#     if not (os.path.isdir(output_dir)):
#         os.makedirs(output_dir)
#     if not (os.path.isdir(trigger_dir)):
#         os.makedirs(trigger_dir)
#     if not (os.path.isdir(ae_dir)):
#         os.makedirs(ae_dir)
#     if not (os.path.isdir(physical_dir)):
#         os.makedirs(physical_dir)
#     if not (os.path.isdir(tbx_dir)):
#         os.makedirs(tbx_dir)


#############################
# Audio Reading Params
#############################
# [16384, 32768, 65536] in case of a longer window change model_capacity_size to 32
window_length = 48000
sampling_rate = 16000
normalize_audio = True
num_channels = 1

#############################
# Logger init
#############################
LOGGER = logging.getLogger("wavegan")
LOGGER.setLevel(logging.DEBUG)
#############################
# Torch Init and seed setting
#############################
cuda = torch.cuda.is_available()
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
# update the seed
manual_seed = 2019
random.seed(manual_seed)
torch.manual_seed(manual_seed)
np.random.seed(manual_seed)
if cuda:
    torch.cuda.manual_seed(manual_seed)
    torch.cuda.empty_cache()
