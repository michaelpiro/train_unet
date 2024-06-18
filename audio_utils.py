import numpy as np
import torch
import torchaudio
import librosa
from all_configurations import TrainingConfig
import matplotlib.pyplot as plt
import librosa.display
from diffusers.utils.torch_utils import randn_tensor

""" =========== Constants =============== """""
SAMPLE_RATE = TrainingConfig.SAMPLE_RATE
N_FFT = TrainingConfig.N_FFT
HOP_LENGTH = TrainingConfig.HOP_LENGTH
WIN_LENGTH = TrainingConfig.WIN_LENGTH
N_MELS = TrainingConfig.N_MELS
FMAX = TrainingConfig.FMAX
FMIN = TrainingConfig.FMIN
AUDIO_LEN_SEC = TrainingConfig.AUDIO_LEN_SEC
TARGET_LENGTH = TrainingConfig.TARGET_MEL_LENGTH
NUM_SAMPLES = TrainingConfig.NUM_SAMPLES
TARGET_LENGTH_SEC = TrainingConfig.TARGET_LENGTH_SEC
# DTYPE = TrainingConfig.mixed_precision
DTYPE = 'torch.float32'

TARGET_SR = int(SAMPLE_RATE / 2)

def load_audio(audio_path):
    """
    Loads an audio file from disk.

    Parameters:
    - audio_path: str, the path to the audio file.
    - sr: int, the sampling rate of the audio file.

    Returns:
    - audio: numpy array, the audio signal.
    - sr: int, the sampling rate of the audio signal.
    """
    audio, sr = torchaudio.load(audio_path, normalize=True)
    audio = to_mono(audio)
    return audio, sr


def save_audio(audio, sr, audio_path):
    """
    Saves an audio signal to disk.

    Parameters:
    - audio: numpy array, the audio signal.
    - sr: int, the sampling rate of the audio signal.
    - audio_path: str, the path to save the audio file.
    """
    torchaudio.save(audio_path, audio, sr)

def to_mono(signal):
    # print(signal.shape[0])

    if signal.shape[0] == 2:
        signal = torch.mean(signal, dim=0, keepdim=False)
    return signal


def extract_log_mel_spectrogram(audio,
                                sr=SAMPLE_RATE,
                                n_fft=N_FFT,
                                hop_length=HOP_LENGTH,
                                n_mels=N_MELS,
                                f_min=FMIN,
                                f_max=FMAX,
                                win_length=WIN_LENGTH
                                ):
    """
    Extracts the log Mel spectrogram from an audio signal.

    Parameters:
    - audio: numpy array, the audio signal.
    - sr: int, the sampling rate of the audio signal.
    - n_fft: int, the length of the FFT window.
    - hop_length: int, the number of samples between successive frames.
    - n_mels: int, the number of Mel bands.

    Returns:
    - log_mel_spectrogram: numpy array of shape (1, 1, 1024, 64), the log Mel spectrogram.
    """
    # Compute Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                                     n_mels=n_mels, fmin=f_min, fmax=f_max, win_length=win_length,
                                                     center=True)
    # Convert to log scale (dB)

    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return log_mel_spectrogram


def crop_audio_randomly(audio1, audio2, length, generator=None):
    """
    Crops the audio tensor to a specified length at a random start index,
    using a PyTorch generator for randomness.

    Parameters:
    - audio_tensor (torch.Tensor): The input audio tensor.
    - length (int): The desired length of the output tensor.
    - generator (torch.Generator, optional): A PyTorch generator for deterministic randomness.

    Returns:
    - torch.Tensor: The cropped audio tensor of the specified length.
    """

    # Ensure the desired length is not greater than the audio tensor length
    # print(length)
    # print(audio1.shape)
    if length > audio1.shape[0]:
        raise ValueError("Desired length is greater than the audio tensor length.")

    # Calculate the maximum start index for cropping
    max_start_index = audio1.size(0) - length

    # Generate a random start index from 0 to max_start_index using the specified generator
    start_index = torch.randint(0, max_start_index + 1, (1,), generator=generator).item()

    # Crop the audio tensor from the random start index to the desired length
    audio1 = audio1[start_index:start_index + length]
    audio2 = audio2[start_index:start_index + length]

    return audio1, audio2


def preprocess(orig_waveform, sr1, no_drums_waveform, sr2, generator=None):
    # original_audio_path = audio_labels.iloc[idx, ORIG_FILE]
    # no_drums_audio_path = audio_labels.iloc[idx, NO_DRUMS_FILE]
    # orig_waveform, sr1 = torchaudio.load(original_audio_path)
    # no_drums_waveform, sr2 = torchaudio.load(no_drums_audio_path)
    # orig_waveform = orig_waveform.half()
    # no_drums_waveform = no_drums_waveform.half()
    if generator is None:
        generator = torch.manual_seed(0)
    # convert to mono
    no_drums_waveform = to_mono(no_drums_waveform).squeeze(0)
    orig_waveform = to_mono(orig_waveform).squeeze(0)

    # print(orig_waveform.shape)
    # print(no_drums_waveform.shape)

    # label = audio_labels.iloc[idx, 2]

    # if the data is not in the correct sample rate, we resample it.
    if TARGET_SR != SAMPLE_RATE:
        orig_waveform = torchaudio.functional.resample(orig_waveform, sr1, TARGET_SR)
        no_drums_waveform = torchaudio.functional.resample(no_drums_waveform, sr2, TARGET_SR)
    else:
        if sr1 != SAMPLE_RATE:
            orig_waveform = torchaudio.functional.resample(orig_waveform, sr1, SAMPLE_RATE)
        if sr2 != SAMPLE_RATE:
            no_drums_waveform = torchaudio.functional.resample(no_drums_waveform, sr2, SAMPLE_RATE)

    orig_waveform, no_drums_waveform = crop_audio_randomly(orig_waveform, no_drums_waveform, NUM_SAMPLES, generator)
    # create log mel spec
    log_mel_orig = torch.from_numpy(extract_log_mel_spectrogram(orig_waveform.numpy()))
    log_mel_no_drums = torch.from_numpy(extract_log_mel_spectrogram(no_drums_waveform.numpy()))
    log_mel_orig = torch.transpose(log_mel_orig, 0, 1)
    log_mel_no_drums = torch.transpose(log_mel_no_drums, 0, 1)
    if DTYPE == 'fp16':
        log_mel_orig = log_mel_orig.half()
        log_mel_no_drums = log_mel_no_drums.half()
    else:
        log_mel_orig = log_mel_orig.to(torch.float32)
        log_mel_no_drums = log_mel_no_drums.to(torch.float32)
    return log_mel_orig, log_mel_no_drums


def show_spectrogram(mel_spectrogram):
    mel_spectrogram = torch.transpose(mel_spectrogram, 0, 1)
    to_np = mel_spectrogram.cpu().numpy()
    # fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.specshow(to_np, x_axis='time', y_axis='mel',
                             sr=TARGET_SR,
                             hop_length=HOP_LENGTH,
                             win_length=WIN_LENGTH)


def transform(examples):
    drums = []
    no_drums = []
    dic = {}
    no_drum, sr1 = torch.tensor(examples["no_drums"][0]["array"]), examples["no_drums"][0]["sampling_rate"]
    drum, sr2 = torch.tensor(examples["drums"][0]["array"], requires_grad=False), examples["drums"][0]["sampling_rate"]
    mel_drums, mel_no_drums = preprocess(drum, sr1, no_drum, sr2)
    no_drums.append(mel_no_drums)
    drums.append(mel_drums)
    dic["no_drums"] = no_drums
    dic["drums"] = drums
    return dic





def crop_audio_randomly_for_inf(audio1, length, generator=None):
    """
    Crops the audio tensor to a specified length at a random start index,
    using a PyTorch generator for randomness.

    Parameters:
    - audio_tensor (torch.Tensor): The input audio tensor.
    - length (int): The desired length of the output tensor.
    - generator (torch.Generator, optional): A PyTorch generator for deterministic randomness.

    Returns:
    - torch.Tensor: The cropped audio tensor of the specified length.
    """

    # Ensure the desired length is not greater than the audio tensor length
    # print(length)
    print(audio1.shape)
    if length > audio1.shape[0]:
        raise ValueError("Desired length is greater than the audio tensor length.")

    # Calculate the maximum start index for cropping
    max_start_index = audio1.size(0) - length

    # Generate a random start index from 0 to max_start_index using the specified generator
    start_index = torch.randint(0, max_start_index + 1, (1,), generator=generator).item()

    # Crop the audio tensor from the random start index to the desired length
    audio1 = audio1[start_index:start_index + length]

    return audio1


def preprocess_for_inferance(orig_waveform, sr1, generator=None):
    if generator is None:
        generator = torch.manual_seed(0)
    # convert to mono
    orig_waveform = to_mono(orig_waveform).squeeze(0)

    # if the data is not in the correct sample rate, we resample it.
    if TARGET_SR != SAMPLE_RATE:
        orig_waveform = torchaudio.functional.resample(orig_waveform, sr1, TARGET_SR)
    else:
        if sr1 != SAMPLE_RATE:
            orig_waveform = torchaudio.functional.resample(orig_waveform, sr1, SAMPLE_RATE)

    orig_waveform = crop_audio_randomly_for_inf(orig_waveform, NUM_SAMPLES, generator)
    # create log mel spec
    log_mel_orig = torch.from_numpy(extract_log_mel_spectrogram(orig_waveform.numpy()))
    log_mel_orig = torch.transpose(log_mel_orig, 0, 1)
    if DTYPE == 'fp16':
        log_mel_orig = log_mel_orig.half()
    else:
        log_mel_orig = log_mel_orig.to(torch.float32)
    return log_mel_orig



def mel_spectrogram_to_waveform(mel_spectrogram,vocoder):
    if mel_spectrogram.dim() == 4:
        print(f"mel_spectrogram shape: {mel_spectrogram.shape}")
        mel_spectrogram = mel_spectrogram.squeeze(1)
    mel_spectrogram = torch.reshape(mel_spectrogram, (mel_spectrogram.shape[1], -1,mel_spectrogram.shape[0]))
    print(f"mel_spectrogram shape: {mel_spectrogram.shape}")
    waveform = vocoder(mel_spectrogram.half())
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    waveform = waveform.cpu().float()
    return waveform

