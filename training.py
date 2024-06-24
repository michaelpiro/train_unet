from accelerate import Accelerator
from huggingface_hub import HfFolder, Repository, whoami
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
import os

import audio_utils
import wandb
import torch.nn.functional as F
from PIL import Image
from audio_utils import mel_spectrogram_to_waveform, preprocess_for_inferance
from diffusers.utils.torch_utils import randn_tensor
import numpy as np

# def diffusion_inference(config, model, vae, noise_scheduler, val_dataloader):

from diffusers import DiffusionPipeline
from diffusers import StableDiffusionPipeline
from diffusers import DDPMPipeline
from diffusers import UNet2DModel
from diffusers import DDIMScheduler
from diffusers import AutoencoderKL
from transformers import SpeechT5HifiGan
import torch


@torch.no_grad()
def inferance(audio, sr, timesteps):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    NEW_REPO_ID = "michaelpiro1/train_model"
    print("Loading models")
    unet = UNet2DModel.from_pretrained("michaelpiro1/unet_repo", subfolder="unet").to(device)
    print(f"num parameters: {unet.num_parameters()}")
    scheduler = DDIMScheduler.from_pretrained(NEW_REPO_ID, subfolder="scheduler")
    scheduler.set_timesteps(timesteps)
    vae = AutoencoderKL.from_pretrained(NEW_REPO_ID, subfolder="vae").to(device)
    vocoder = SpeechT5HifiGan.from_pretrained(NEW_REPO_ID, subfolder="vocoder")
    print(f"preprocessing audio, audio shape: {audio.shape}")
    log_mel = preprocess_for_inferance(audio, sr).unsqueeze(0).unsqueeze(0)
    log_mel = log_mel.to(device)
    print(f"audio shape: {log_mel.shape}")
    print("encoding spectogram to latent space")
    clean_latent_no_drums = vae.encode(log_mel).latent_dist.sample()
    # Sample noise to add to the images
    # noise = torch.randn(clean_images_no_drums.shape).to(clean_images_no_drums.device)
    # bs = clean_latent_no_drums.shape[0]
    latent = randn_tensor(clean_latent_no_drums.shape, generator=None,
                                          device=vae.device, dtype=clean_latent_no_drums.dtype)

    # Add noise to the clean images according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    # noisy_images = noise_scheduler.add_noise(clean_images_drums, noise, timesteps)
    print("diffusing noise")
    for i in tqdm(range(timesteps)):
        inp = torch.cat([clean_latent_no_drums, latent], dim=1)
        noise_pred = unet(inp, i, return_dict=False)[0]
        latent = scheduler.step(noise_pred, i, latent).prev_sample
    print("decoding latent space to mel spectogram")
    latents = 1 / vae.config.scaling_factor * latent
    mel_spectrogram = vae.decode(latents).sample

    # mel to waveform
    if mel_spectrogram.dim() == 4:
        print(f"mel_spectrogram shape: {mel_spectrogram.shape}")
        mel_spectrogram = mel_spectrogram.squeeze(1)
    mel_spectrogram = torch.reshape(mel_spectrogram, (mel_spectrogram.shape[0], mel_spectrogram.shape[1], -1))
    print(f"final size mel_spectrogram shape: {mel_spectrogram.shape}")
    # mel_spectrogram = (vae.decode(clean_latent_no_drums).sample).squeeze(0)
    waveform = vocoder(mel_spectrogram)
    # waveform = vocoder(log_mel.squeeze(0))
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    waveform = waveform.cpu().float()
    # audio = mel_spectrogram_to_waveform(mel_spectrogram)

    return waveform, audio_utils.TARGET_SR


@torch.no_grad()
def evaluate(config, model, vae, noise_scheduler, val_dataloader, global_step=0):
    # """Evaluate the model on the validation set, and save the outputs in a local directory."""
    # # Initialize accelerator
    # accelerator = Accelerator()
    # if accelerator.is_main_process:
    #     os.makedirs(config.output_dir, exist_ok=True)
    #
    # # Prepare everything
    # model, vae, val_dataloader = accelerator.prepare(model, vae, val_dataloader)
    # model.eval()
    # vae.eval()
    #
    # # Evaluate the model
    # for step, batch in enumerate(val_dataloader):
    #     original = batch["original"].unsqueeze(0)
    #     # clean_images_drums = vae.encode(clean_images_drums).latent_dist.sample()
    #     clean_images_no_drums = batch["no_drums"].unsqueeze(0)
    #     clean_latent_no_drums = vae.encode(clean_images_no_drums).latent_dist.sample()
    #     # Forward pass
    #     device = clean_images_no_drums.device
    #     latent = randn_tensor(clean_latent_no_drums.shape, generator=None,
    #                                 device=device, dtype=clean_latent_no_drums.dtype) \
    #              * noise_scheduler.init_noise_sigma
    #     for t in range(config.num_diffusion_timesteps):
    #         inp = torch.cat((clean_images_no_drums, latent), dim=1)
    #         noise_pred = model(inp, t, return_dict=False)[0]
    #         latent = noise_scheduler.step(noise_pred, t, latent).prev_sample
    #         # Compute the loss
    #     # save the output to the output directory
    #     output = vae.decode(latent).sample
    #     output = output.squeeze(0)
    #     output = output.cpu().numpy()
    #     output = (output * 255).astype(np.uint8)
    #     output = Image.fromarray(output)
    #     output.save(os.path.join(config.output_dir, f"global_{global_step}_step{step}.png"))
    pass


def train_loop(config, model, vae, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="wandb",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        # if config.push_to_hub:
        #     repo_id = create_repo(
        #         repo_id=config.hub_model_id or Path(config.output_dir).name, exist_ok=True
        #     ).repo_id
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, vae, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, vae, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        scaler = torch.cuda.amp.GradScaler()  # Initialize GradScaler for mixed precision training

        for step, batch in enumerate(train_dataloader):
            #  batch["drums"] is all the spectograms of the songs with drums
            # clean_images_drums = batch["drums"].unsqueeze(0)
            # print(batch)
            # print(batch[0].shape)
            clean_images_drums,clean_images_no_drums = batch
            # print(clean_images_drums.shape)
            clean_images_no_drums = clean_images_no_drums.unsqueeze(0)
            clean_images_no_drums = clean_images_no_drums.reshape(-1,1,clean_images_no_drums.shape[2],clean_images_no_drums.shape[3])
            clean_images_drums = clean_images_drums.unsqueeze(0)
            clean_images_drums = clean_images_drums.reshape(-1,1,clean_images_drums.shape[2],clean_images_drums.shape[3])
            print(clean_images_drums.shape)
            # print(vae.device,model.device)

            # print(clean_images_drums.shape)
            clean_images_drums = vae.encode(clean_images_drums).latent_dist.sample()
            # clean_images_no_drums = batch["no_drums"].unsqueeze(0)
            clean_images_no_drums = vae.encode(clean_images_no_drums).latent_dist.sample()
            # Sample noise to add to the images
            noise = torch.randn(clean_images_drums.shape).to(clean_images_drums.device)
            bs = clean_images_drums.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images_drums.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images_drums, noise, timesteps)

            # with accelerator.accumulate(model):
            with accelerator.accumulate(model):
                # Predict the noise residual
                # noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                inp = torch.cat([clean_images_no_drums, noisy_images], dim=1)
                noise_pred = model(inp, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    pipeline.push_to_hub(config.TRAINING_REPO, commit_message=f"Epoch {epoch}")
                pipeline.save_pretrained(config.output_dir)

