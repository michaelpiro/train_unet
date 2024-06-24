# from diffusers import UNet2DModel
# import scipy
# import torch
# import librosa
#
import torch
import wandb
from diffusers import AudioLDM2Pipeline

SPEC_LENGTH = 128
SPEC_RESOLUTION = 64
BATCH_SIZE = 1
LATENT_CHANNELS = 8
ENCODED_LENGTH = 128 // 4
ENCODED_RESOLUTION = 64 // 4

# repo_id = "cvssp/audioldm2"
# pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
# pipe = pipe.to("cuda")
# vae = pipe.vae


# rand_vector_shape = (1,1,128,64)
# encoded_shape = (1,8,128//4,64//4)
# print(encoded_shape)

# rand_vec = torch.randn(1,1,128,64,dtype=torch.float16)
# define the prompts
# prompt = "The sound of a hammer hitting a wooden surface."
# negative_prompt = "Low quality."
# encoded = vae.encode(rand_vec).latent_dist.sample()
# print(encoded.shape)

# set the seed for generator
# generator = torch.Generator("cuda").manual_seed(0)

# run the generation
# audio = pipe(
#     prompt,
#     negative_prompt=negative_prompt,
#     num_inference_steps=200,
#     audio_length_in_s=10.0,
#     num_waveforms_per_prompt=3,
#     generator=generator,
# ).audios

# save the best audio sample (index 0) as a .wav file
# scipy.io.wavfile.write("techno.wav", rate=16000, data=audio[0])


# SPEC_LENGTH = 128
# SPEC_RESOLUTION = 64


# model = UNet2DModel(
#     sample_size=ENCODED_LENGTH*ENCODED_RESOLUTION,  # the target image resolution
#     in_channels=16,  # the number of input channels, 3 for RGB images
#     out_channels=8,  # the number of output channels
#     layers_per_block=2,  # how many ResNet layers to use per UNet block
#     # block_out_channels=(128, 256, 512, 1024),  # the number of output channels for each UNet block
#     block_out_channels=(128, 256, 384, 640),  # the number of output channels for each UNet block
#
#     down_block_types=(
#         "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
#         "AttnDownBlock2D",
#         "AttnDownBlock2D",
#         "DownBlock2D",  # a regular ResNet downsampling block
#     ),
#     up_block_types=(
#         "UpBlock2D",
#         "AttnUpBlock2D",  # a regular ResNet upsampling block
#         "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
#         "AttnUpBlock2D",
#
#     ),
# )


# print(f"num params :{model.num_parameters()/1e6}M")
# # model.half()
# rand_vec = torch.randn(encoded_shape,dtype=torch.float32)
# model_input = torch.cat([rand_vec,rand_vec],dim=1)
# print(f"model input shape :{model_input.shape}")
# output = model(model_input,1,return_dict=False)[0]
# print(f"output shape :{output.shape}")


import datasets as ds
# dataset = ds.load_dataset("audiofolder", data_dir="/Users/mac/Desktop/demucs_out/mdx_extra")
# print(dataset["train"][0]["audio"])
# print(dataset["train"][1])
# print(dataset["train"][2])
# print(dataset["train"][3])
import os

path_to_file_dir = os.path.dirname("/Users/mac/Desktop/demucs_out/mdx_extra")
print(path_to_file_dir)


def create_dataset_dict(data_dir):
    dataset = ds.load_dataset("audiofolder", data_dir=data_dir)
    dataset_dict = {}
    j = 0
    drums = []
    no_drums = []
    for i in dataset["train"]:
        if j % 2 == 0:
            drums.append(i["audio"]["path"])
        else:
            no_drums.append(i["audio"]["path"])
        j += 1
        # drums = dataset["train"][i]["audio"]
        # no_drums = dataset["train"][i+1]["audio"]

        # print(p1)
        # p2 = path2.append(dataset["train"][i+1])
        # sr1 = dataset["train"][i]["sampling_rate"]
        # sr2 = dataset["train"][i+1]["sampling_rate"]
        # path_to_file_dir = os.path.dirname(path1)
        # audio = drums,no_drums
        # sr = sr1,sr2

        # dataset_dict[i] = {"audio":audio,"label":label}
    dataset_dict["drums"] = drums
    dataset_dict["no_drums"] = no_drums
    # audio_dataset = ds.Dataset.from_dict(dataset_dict).cast_column(("drums","no_drums"),ds.Audio())
    # print(audio_dataset[0])
    # print(audio_dataset[1])
    return dataset_dict


#
# d = create_dataset_dict(data_dir="/Users/mac/Desktop/demucs_out/mdx_extra")
# audio_dataset = ds.Dataset.from_dict(d).cast_column("drums",ds.Audio())
# audio_dataset = audio_dataset.cast_column("no_drums",ds.Audio())
# audio_dataset.push_to_hub("michaelpiro1/separated_music")

NEW_REPO_ID = "michaelpiro1/train_model"
from transformers import SpeechT5HifiGan
# vocoder = SpeechT5HifiGan.from_pretrained(NEW_REPO_ID ,subfolder= "vocoder")
# log_mel = torch.rand(1,256,64)
# audio = vocoder(log_mel)
# print(audio.shape)
# print(40992/16000)

# print(audio_dataset[0])
# print(audio_dataset[1])

# create_dataset_dict("/Users/mac/Desktop/demucs_out/mdx_extra")

# {'audio': {'path': '/Users/mac/Desktop/demucs_out/mdx_extra/012045/drums.mp3', 'array': array([0.00000000e+00, 2.75437659e-13, 2.28134135e-12, ...,
#        3.24447220e-02, 3.34393485e-02, 3.39863487e-02]), 'sampling_rate': 44100}, 'label': 0}
from huggingface_hub import notebook_login, create_repo, hf_hub_download, login
from diffusers import DiffusionPipeline
from diffusers import StableDiffusionPipeline
from diffusers import UNet2DModel
from diffusers import DDIMScheduler
from diffusers import AutoencoderKL
from transformers import SpeechT5HifiGan
from diffusers import MusicLDMPipeline

from datasets import load_dataset
import torch
import audio_utils
from all_configurations import TrainingConfig

from dataclasses import dataclass
from torch.utils.data import DataLoader
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup

def load_dataloader(config):
    config.dataset_name = "michaelpiro1/separated_music"
    dataset = load_dataset(config.dataset_name, split="train")
    dataset.set_transform(audio_utils.transform)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)
    return train_dataloader

from diffusers.optimization import get_cosine_schedule_with_warmup


def load_models(config, device=torch.device("cpu")):
    NEW_REPO_ID = config.NEW_REPO_ID
    unet = UNet2DModel.from_pretrained(NEW_REPO_ID, subfolder="unet").to(device)
    scheduler = DDIMScheduler.from_pretrained(NEW_REPO_ID, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(NEW_REPO_ID, subfolder="vae").to(device)
    vocoder = SpeechT5HifiGan.from_pretrained(NEW_REPO_ID, subfolder="vocoder").to(device)
    return unet, vae, scheduler, vocoder


import training as tr
import torch.nn.functional as F
import torchaudio
from dataset import dataset

if __name__ == '__main__':
    config = TrainingConfig()
    login(token=config.HF_TOKEN)
    config.csv_file_path = "C:\\Users\\michaelpiro1\\PycharmProjects\\train_unet\\dataset\\file.csv"
    dataset = dataset.CustomAudioDataset(config.csv_file_path)
    # d = create_dataset_dict("D:\\yuval.shaffir\\separated\\mdx_extra")
    # audio_dataset = ds.Dataset.from_dict(d).cast_column("drums", ds.Audio())
    # audio_dataset = audio_dataset.cast_column("no_drums", ds.Audio())
    # audio_dataset.push_to_hub("michaelpiro1/separated_music")
    wandb.login(key=config.wandb)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(device)
    # train_data_loader = load_dataloader(config)
    train_data_loader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)
    unet, vae, scheduler, vocoder = load_models(config, device=device)
    unet.train()
    vae.eval()
    vocoder.eval()
    optimizer = torch.optim.AdamW(unet.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_data_loader) * config.num_epochs),
    )
    tr.train_loop(config, unet, vae, scheduler, optimizer, train_data_loader,lr_scheduler)
    # audio, sr = audio_utils.load_audio("audio_example/no_drums.mp3")
    # print(f"audio shape: {audio.shape}")
    # resampled = torchaudio.functional.resample(audio, sr, audio_utils.TARGET_SR)
    # cropped_audio = resampled[10*sr:config.NUM_SAMPLES +10*sr]
    # new_audio,new_sr = tr.inferance(cropped_audio, audio_utils.TARGET_SR, 50)
    # # print(f"copped audio shape: {cropped_audio.shape}")
    # # print(f"new audio length: {len(new_audio)}")
    # audio_utils.save_audio(new_audio, new_sr,os.path.join(config.output_dir, "no_drums_new.mp3"))
    # TRAINING_REPO = "michaelpiro1/unet_repo"
    # HF_TOKEN = "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"
    # NEW_REPO_ID = "michaelpiro1/train_model"
