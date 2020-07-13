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
    '''Apply random resize along time dimention to spectrogram.'''
    def __init__(self, resize_range=0.2, resize_mode='billinear'):
        self.resize_range = resize_range
        self.resize_mode = resize_mode

    def __call__(self, spec):
        scale_factor = 1 + (np.random.random_sample() - 0.5) * self.resize_range
        new_spec = nn.Upsample(
            scale_factor=(1, scale_factor),
            mode=self.resize_mode,
            align_corners=False
        )(
            spec.unsqueeze(0).unsqueeze(0)
        ).squeeze(0).squeeze(0)
        return new_spec


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
        return new_spec


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


class SpecMixup:
    '''
    Mixup spectrogram of a sample with a spectrogram of a random sample from the mixup dataset.
    The smallest spectrogram is repeated until both spectograms are of the same size.
    '''
    def __init__(self, mixup_sample_dataset, alpha=0.5, mixup_labels=True):
        self.mixup_sample_dataset = mixup_sample_dataset
        self.alpha = alpha
        self.mixup_labels = mixup_labels

    def __call__(self, sample):
        i = np.random.randint(0, len(self.mixup_sample_dataset))
        mixup_sample = self.mixup_sample_dataset[i]

        sample_spec = sample['mel_spec']
        mixup_sample_spec = mixup_sample['mel_spec']

        sample_len = sample_spec.size(1)
        mixup_sample_len = mixup_sample_spec.size(1)

        if sample_len > mixup_sample_len:
            k = int(np.ceil(sample_len / mixup_sample_len))
            mixup_sample_spec = mixup_sample_spec.repeat(1, k)[:, :sample_len]
        elif mixup_sample_len > sample_len:
            k = int(np.ceil(mixup_sample_len / sample_len))
            sample_spec = sample_spec.repeat(1, k)[:, :mixup_sample_len]

        result_spec = sample_spec * (1 - self.alpha) + mixup_sample_spec * self.alpha

        if self.mixup_labels:
            result_encoded_ebird_codes = (
                (sample['encoded_ebird_codes'] == 1.) | (mixup_sample['encoded_ebird_codes'] == 1.)
            ).float()
        else:
            result_encoded_ebird_codes = sample['encoded_ebird_codes']

        sample['encoded_ebird_codes'] = result_encoded_ebird_codes
        sample['mel_spec'] = result_spec

        return sample


class SpecTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        sample['mel_spec'] = self.transform(sample['mel_spec'])
        return sample
