import random

import numpy as np
import scipy as sp
import soundfile as sf
import librosa
from torch.utils.data import Dataset


SR = 16000
DEMAND_SR = 48000
SR_RATIO = DEMAND_SR // SR
WINDOW_STEP = 256
WINDOW_SIZE = WINDOW_STEP * 2
N_CHANS = 512
TRACK_LEN_LIMIT = 1000 * WINDOW_STEP


def wav_to_sg(signal, calc_phase=False):
    stft = librosa.stft(signal, n_fft=N_CHANS, hop_length=WINDOW_STEP, win_length=WINDOW_SIZE,
                        window=sp.signal.hamming)
    sg = np.abs(stft)
    if calc_phase:
        phase = np.angle(stft)
        return sg, phase
    return sg


def sg_to_wav(sg, phase):
    return librosa.istft(sg * np.exp(1j * phase), hop_length=WINDOW_STEP, win_length=WINDOW_SIZE,
                         window=sp.signal.hamming)


def adjust_snr(signal, noise, target_snr):
    signal_amp = np.mean(signal ** 2) ** .5
    noise_amp = np.mean(noise ** 2) ** .5
    target_noise_amp = signal_amp / 10 ** (target_snr / 10)
    return noise * (target_noise_amp / noise_amp)


class DemandNoiser:
    def __init__(self, noise_dirs, snr_options):
        self.noises = [
            sp.io.wavfile.read(str(noise_path))[1][::SR_RATIO].copy()
            for noise_dir in noise_dirs
            for noise_path in noise_dir.glob('*.wav')
        ]
        self.snr_options = snr_options

    def add_noise(self, track, track_id=None):
        # By default, random noise is added with random signal-to-noise ratio.
        # Passing track identifier enables fixed noise sample and fixed SNR for the track.
        # This is done to get diverse training data and stable validation and test results.
        if track_id is not None:
            noise = self.noises[track_id % len(self.noises)][:len(track)]
            snr = self.snr_options[track_id % len(self.snr_options)]
        else:
            noise = random.choice(self.noises)[:len(track)]
            snr = random.choice(self.snr_options)
        return track + adjust_snr(track, noise, snr)


class LibreSpeechDataset(Dataset):
    def __init__(self, dirs, noiser, random_noise=True):
        super().__init__()
        self.track_paths = [
            str(track_path)
            for tracks_dir in dirs
            for track_path in tracks_dir.glob('**/*.flac')
        ]
        self.noiser = noiser
        self.random_noise = random_noise

    def __getitem__(self, index):
        track, _ = sf.read(self.track_paths[index])
        track = track[:TRACK_LEN_LIMIT]
        noised_track = self.noiser.add_noise(track, None if self.random_noise else index)
        return wav_to_sg(noised_track, True), wav_to_sg(track, True)

    def __len__(self):
        return len(self.track_paths)
