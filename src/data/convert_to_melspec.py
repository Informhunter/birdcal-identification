import os
import click
import gc
import torchaudio as toa
import torch as t
from tqdm import tqdm
from pathlib import Path


@click.command()
@click.argument('input_dir')
@click.argument('output_dir')
@click.option('--resample_rate', default=44100)
@click.option('--n_fft', default=2048)
@click.option('--n_mels', default=128)
@click.option('--hop_length', default=512)
@click.option('--use_one_channel', is_flag=True, default=True)
def main(input_dir, output_dir, resample_rate, n_fft, n_mels, hop_length, use_one_channel):

    if not use_one_channel:
        raise NotImplementedError('Can only use one channel')

    f_paths = Path(input_dir).rglob('*.mp3')

    resample_transforms = {}
    mel_transform = toa.transforms.MelSpectrogram(
        sample_rate=resample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        hop_length=hop_length,
    )

    for f_path in tqdm(f_paths):

        try:
            waveform, old_sampling_rate = toa.load(f_path)
        except RuntimeError:
            print(f'Failed to load: {f_path}')
            continue

        waveform = waveform[0]

        with t.no_grad():
            if old_sampling_rate not in resample_transforms:
                resample_transforms[old_sampling_rate] = toa.transforms.Resample(
                    old_sampling_rate,
                    resample_rate
                )

            resample_transform = resample_transforms[old_sampling_rate]
            waveform = resample_transform(waveform)
            mel_spec = mel_transform(waveform)

        f_name = os.path.basename(f_path)
        new_f_name = os.path.splitext(f_name)[0] + '.pt'
        ebird_code = os.path.basename(os.path.dirname(f_path))
        new_dir = os.path.join(output_dir, ebird_code)
        new_f_path = os.path.join(new_dir, new_f_name)
        Path(new_dir).mkdir(parents=True, exist_ok=True)

        t.save(mel_spec, new_f_path)

        gc.collect()


if __name__ == '__main__':
    main()
