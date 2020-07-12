import torch as t
import numpy as np
import torch.nn as nn


class RandomCrop:
    '''Select random piece of spectrogram with fixed duration.'''
    def __init__(self, n_frames):
        self.n_frames = n_frames

    def __call__(self, spec):
        duration = spec.size(1)
        if duration > self.n_frames:
            max_i = duration - self.n_frames
            i = np.random.randint(0, max_i)
            spec = spec[i, i + self.n_frames]

        return spec


class RandomTimeResize:
    '''Aplly random resize along time dimention to spectrogram.'''
    def __init__(self, resize_range=0.2, resize_mode='billinear'):
        self.resize_range = resize_range
        self.resize_mode = resize_mode

    def __call__(self, spec):
        scale_factor = 1 + (np.random.random_sample() - 0.5) * self.resize_range
        mel_spec = nn.Upsample(
            scale_factor=(1, scale_factor),
            mode=self.resize_mode,
            align_corners=False
        )(
            spec.unsqueeze(0).unsqueeze(0)
        ).squeeze(0).squeeze(0)
        return mel_spec


class RandomTimeShift:
    '''Apply random cyclical shift to spectrogram.'''
    def __init__(self, shift_range=0.5):
        self.shift_range = shift_range

    def __call__(self, spec):
        new_spec = t.zeros_like(spec)
        duration = spec.size(1)
        shift = int(np.random.random_sample() * duration * self.shift_range) + 1
        new_spec[:, :shift] = spec[:, -shift:]
        new_spec[:, shift:] = spec[:, :-shift]
        spec = new_spec

        return spec


class WaveformRandomTimeResize:
    '''Aplly random resize along time dimention to waveform.'''
    def __init__(self, resize_range=0.2, resize_mode='linear'):
        self.resize_range = resize_range
        self.resize_mode = resize_mode

    def __call__(self, waveform):
        scale_factor = 1 + (np.random.random_sample() - 0.5) * self.resize_range
        waveform = nn.Upsample(
            scale_factor=(scale_factor,),
            mode=self.resize_mode,
            align_corners=False
        )(
            waveform.unsqueeze(0).unsqueeze(0)
        ).squeeze(0).squeeze(0)
        return waveform


class WaveformRandomTimeShift:
    '''Apply random cyclical shift to waveform.'''
    def __init__(self, shift_range=0.5):
        self.shift_range = shift_range

    def __call__(self, waveform):
        new_waveform = t.zeros_like(waveform)
        duration = waveform.size(0)
        shift = int(np.random.random_sample() * duration * self.shift_range) + 1
        new_waveform[:shift] = waveform[-shift:]
        new_waveform[shift:] = waveform[:-shift]

        return new_waveform
