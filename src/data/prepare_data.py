'''
Prepare dataset for training:
1. Resample audio
2. Split each recording into pieces so that each piece is shorter.
3. Build and save mel spectrograms.
4. Create a custom train.csv file describing the new dataset.

file.mp3 -> file_part_1.mp3, file_part_2.mp3, ..
'''

import os
import click
import torch as t
import torchaudio as toa
from pathlib import Path
from joblib import Parallel, delayed


def resample_waveform(waveform, old_sampling_rate, target_sampling_rate):
    resample_transform = toa.transforms.Resample(
        old_sampling_rate,
        target_sampling_rate
    )
    waveform = resample_transform(waveform)
    return waveform


def split_waveform(waveform, sampling_rate=44100, max_duration=60):
    assert len(waveform.size()) == 1  # No channels
    max_duration_n = sampling_rate * max_duration
    duration_n = waveform.size(0)
    if duration_n <= max_duration:
        waveforms = [waveform]
    else:
        k = duration_n // max_duration_n + 1
        piece_duration_n = duration_n // k
        waveforms = []
        current_n = 0
        for i in range(k-1):
            waveforms.append(waveform[current_n:current_n+piece_duration_n])
            current_n += piece_duration_n
        waveforms.append(waveform[current_n:])
    return waveforms


def create_mel_spec(waveform, sampling_rate, n_fft, n_mels, hop_length):
    mel_transform = toa.transforms.MelScale(
        sample_rate=sampling_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        hop_length=hop_length
    )
    mel_spec = mel_transform(waveform)
    return mel_spec


def process_file(f_path, output_dir, target_sampling_rate, max_duration,
                 n_fft, n_mels, hop_length):
    try:
        waveform, old_sampling_rate = toa.load(f_path)
    except RuntimeError:
        print(f'Failed to load: {f_path}')
        return

    waveform = resample_waveform(waveform, old_sampling_rate, target_sampling_rate)
    waveforms = split_waveform(waveform, target_sampling_rate, max_duration)
    mel_specs = [create_mel_spec(x, n_fft, n_mels, hop_length) for x in waveforms]

    f_name = os.path.basename(f_path)
    new_f_name = os.path.splitext(f_name)[0]
    ebird_code = os.path.basename(os.path.dirname(f_path))
    new_dir = os.path.join(output_dir, ebird_code)

    for i, mel_spec in enumerate(mel_specs):
        new_f_path = os.path.join(new_dir, f'{new_f_name}_part_{i}.pt')
        Path(new_dir).mkdir(parents=True, exist_ok=True)
        t.save(mel_spec, new_f_path)


@click.command()
@click.argument('input_dir', type=click.Path())
@click.argument('output_dir', type=click.Path())
@click.option('--target_sampling_rate', default=44100)
@click.option('--max_duration', default=60)
@click.option('--n_fft', default=2048)
@click.option('--n_mels', default=128)
@click.option('--hop_length', default=512)
@click.option('--n_jobs', default=28)
def main(input_dir, output_dir, target_sampling_rate,
         max_duration, n_fft, n_mels, hop_length, n_jobs):
    f_paths = Path(input_dir).rglob('*.mp3')
    Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_file)(
            f_path, output_dir, target_sampling_rate,
            max_duration, n_fft, n_mels, hop_length
        )
        for f_path in f_paths
    )


if __name__ == '__main__':
    main()
