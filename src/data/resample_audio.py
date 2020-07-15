import os
import click
import torchaudio as toa
import torch as t
from pathlib import Path
from joblib import Parallel, delayed


def process_file(f_path, resample_rate, output_dir):
    try:
        waveform, old_sampling_rate = toa.load(f_path)
    except RuntimeError:
        print(f'Failed to load: {f_path}')
        return

    waveform = waveform[0]

    with t.no_grad():
        resample_transform = toa.transforms.Resample(
            old_sampling_rate,
            resample_rate
        )
        waveform = resample_transform(waveform)

    f_name = os.path.basename(f_path)
    new_f_name = os.path.splitext(f_name)[0] + '.mp3'
    ebird_code = os.path.basename(os.path.dirname(f_path))
    new_dir = os.path.join(output_dir, ebird_code)
    new_f_path = os.path.join(new_dir, new_f_name)
    Path(new_dir).mkdir(parents=True, exist_ok=True)

    toa.save(new_f_path, waveform, resample_rate)


@click.command()
@click.argument('input_dir')
@click.argument('output_dir')
@click.option('--resample_rate', default=44100)
@click.option('--n_jobs', default=30)
def main(input_dir, output_dir, resample_rate, n_jobs):
    f_paths = Path(input_dir).rglob('*.mp3')
    Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_file)(f_path, resample_rate, output_dir)
        for f_path in f_paths
    )


if __name__ == '__main__':
    main()
