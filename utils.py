import os
import time
import math
import torch
import random
import librosa
import librosa.display
import numpy as np
from torch.utils import data
import matplotlib
import matplotlib.pyplot as plt
import glob
import pescador
import torch.nn as nn
from torch.autograd import Variable
from params import *

#############################
# File Utils
#############################
def get_recursive_files(folderPath, ext):
    results = os.listdir(folderPath)
    outFiles = []
    for file in results:
        if os.path.isdir(os.path.join(folderPath, file)):
            outFiles += get_recursive_files(os.path.join(folderPath, file), ext)
        elif file.endswith(ext):
            outFiles.append(os.path.join(folderPath, file))

    return outFiles


def make_path(output_path):
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    return output_path


#############################
# Plotting utils
#############################
def visualize_audio(audio_tensor, is_monphonic=False):
    # takes a batch ,n channels , window length and plots the spectogram
    input_audios = audio_tensor.detach().cpu().numpy()
    plt.figure(figsize=(18, 50))
    for i, audio in enumerate(input_audios):
        plt.subplot(10, 2, i + 1)
        if is_monphonic:
            plt.title("Monophonic %i" % (i + 1))
            librosa.display.waveplot(audio[0], sr=sampling_rate)
        else:
            D = librosa.amplitude_to_db(np.abs(librosa.stft(audio[0])), ref=np.max)
            librosa.display.specshow(D, y_axis="linear")
            plt.colorbar(format="%+2.0f dB")
            plt.title("Linear-frequency power spectrogram %i" % (i + 1))
    if not (os.path.isdir("visualization")):
        os.makedirs("visualization")
    plt.savefig("visualization/interpolation.png")


def visualize_loss(loss_1, loss_2, first_legend, second_legend, y_label):
    plt.figure(figsize=(10, 5))
    plt.title("{} and {} Loss During Training".format(first_legend, second_legend))
    plt.plot(loss_1, label=first_legend)
    plt.plot(loss_2, label=second_legend)
    plt.xlabel("iterations")
    plt.ylabel(y_label)
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()
    if not (os.path.isdir("visualization")):
        os.makedirs("visualization")
    plt.savefig("visualization/loss.png")


def latent_space_interpolation(model, n_samples=10):
    z_test = sample_noise(2)
    with torch.no_grad():
        interpolates = []
        for alpha in np.linspace(0, 1, n_samples):
            interpolate_vec = alpha * z_test[0] + ((1 - alpha) * z_test[1])
            interpolates.append(interpolate_vec)

        interpolates = torch.stack(interpolates)
        generated_audio = model(interpolates)
    visualize_audio(generated_audio, True)


#############################
# Wav files utils
#############################
# Fast loading used with wav files only of 8 bits
def load_wav(wav_file_path):
    try:
        audio_data, _ = librosa.load(wav_file_path, sr=sampling_rate)

        if normalize_audio:
            # Clip magnitude
            max_mag = np.max(np.abs(audio_data))
            if max_mag > 1:
                audio_data /= max_mag
    except Exception as e:
        LOGGER.error("Could not load {}: {}".format(wav_file_path, str(e)))
        raise e
    audio_len = len(audio_data)
    if audio_len < window_length:
        pad_length = window_length - audio_len
        left_pad = pad_length // 2
        right_pad = pad_length - left_pad
        audio_data = np.pad(audio_data, (left_pad, right_pad), mode="constant")

    return audio_data.astype("float32")


def sample_audio(audio_data, start_idx=None, end_idx=None):
    audio_len = len(audio_data)
    if audio_len == window_length:
        # If we only have a single 1*window_length audio, just yield.
        sample = audio_data
    else:
        # Sample a random window from the audio
        if start_idx is None or end_idx is None:
            start_idx = np.random.randint(0, (audio_len - window_length) // 2)
            end_idx = start_idx + window_length
        sample = audio_data[start_idx:end_idx]
    sample = sample.astype("float32")
    assert not np.any(np.isnan(sample))
    return sample, start_idx, end_idx


def sample_buffer(buffer_data, start_idx=None, end_idx=None):
    audio_len = len(buffer_data) // 4
    if audio_len == window_length:
        # If we only have a single 1*window_length audio, just yield.
        sample = buffer_data
    else:
        # Sample a random window from the audio
        if start_idx is None or end_idx is None:
            start_idx = np.random.randint(0, (audio_len - window_length) // 2)
            end_idx = start_idx + window_length
        sample = buffer_data[start_idx * 4 : end_idx * 4]
    return sample, start_idx, end_idx


def wav_generator(file_path):
    audio_data = load_wav(file_path)
    while True:
        sample, _, _ = sample_audio(audio_data)
        yield {"single": sample}


def create_stream_reader(single_signal_file_list):
    data_streams = []
    for audio_path in single_signal_file_list:
        stream = pescador.Streamer(wav_generator, audio_path)
        data_streams.append(stream)
    mux = pescador.ShuffledMux(data_streams)
    batch_gen = pescador.buffer_stream(mux, batch_size)
    return batch_gen


def save_samples(epoch_samples, epoch):
    """
    Save output samples.
    """
    sample_dir = make_path(os.path.join(output_dir, str(epoch)))

    for idx, sample in enumerate(epoch_samples):
        output_path = os.path.join(sample_dir, "{}.wav".format(idx + 1))
        sample = sample[0]
        librosa.output.write_wav(output_path, sample, sampling_rate)


#############################
# Sampling from model
#############################
def sample_noise(size):
    z = torch.FloatTensor(size, noise_latent_dim).to(device)
    z.data.normal_()  # generating latent space based on normal distribution
    return z


#############################
# Model Utils
#############################


def update_optimizer_lr(optimizer, lr, decay):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr * decay


def gradients_status(model, flag):
    for p in model.parameters():
        p.requires_grad = flag


def weights_init(m):
    if isinstance(m, nn.Conv1d):
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        m.bias.data.fill_(0)


#############################
# Creating Data Loader and Sampler
#############################
class WavDataLoader:
    def __init__(self, folder_path, audio_extension="wav"):
        self.signal_paths = get_recursive_files(folder_path, audio_extension)
        self.data_iter = None
        self.initialize_iterator()

    def initialize_iterator(self):
        data_iter = create_stream_reader(self.signal_paths)
        self.data_iter = iter(data_iter)

    def __len__(self):
        return len(self.signal_paths)

    def numpy_to_tensor(self, numpy_array):
        numpy_array = numpy_array[:, np.newaxis, :]
        return torch.Tensor(numpy_array).to(device)

    def __iter__(self):
        return self

    def __next__(self):
        x = next(self.data_iter)
        return self.numpy_to_tensor(x["single"])


if __name__ == "__main__":
    # For debugging purposes
    import time 
    start = time.time()
    print(time.time() - start)
    train_loader = WavDataLoader(os.path.join("piano", "train"), "wav")
    start = time.time()
    for i in range(7):
        x = next(train_loader)
    print(time.time() - start)
