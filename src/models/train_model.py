import click
import random
import numpy as np
import pandas as pd
import torch as t
import pytorch_lightning as pl
import warnings

from src.models.simple_cnn import SimpleCNN, Collate
from src.data.dataset import BirdMelTrainDataset
from src.data.transforms import RandomTimeShift, RandomTimeResize, SpecMixup, SpecTransform
from torchaudio.transforms import TimeMasking, FrequencyMasking
from torchvision.transforms import Compose, RandomApply

from sklearn.model_selection import train_test_split


def prepare_datasets(meta_path, mels_dir, random_state=123):
    df = pd.read_csv(meta_path)
    df = df[df['filename'] != 'XC313679.mp3']
    df = df[df['duration'] < 245]
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df['ebird_code'],
        random_state=random_state
    )
    train_mixup_dataset = BirdMelTrainDataset(
        train_df,
        mels_dir,
        True,
        Compose([
            RandomApply([SpecTransform(RandomTimeShift(1.0))], 0.8),
            RandomApply([SpecTransform(RandomTimeResize(resize_mode='bilinear'))], 0.5),
        ])
    )
    train_dataset = BirdMelTrainDataset(
        train_df,
        mels_dir,
        True,
        Compose([
            RandomApply([SpecMixup(train_mixup_dataset, 0.5, True)], 0.3),
            RandomApply([SpecTransform(RandomTimeShift(1.0))], 0.8),
            RandomApply([SpecTransform(RandomTimeResize(resize_mode='bilinear'))], 0.5),
            SpecTransform(TimeMasking(10)),
            SpecTransform(FrequencyMasking(8)),
        ])
    )

    test_dataset = BirdMelTrainDataset(
        test_df,
        mels_dir,
        True
    )

    return train_dataset, test_dataset


@click.command()
def main():

    warnings.filterwarnings('ignore')  # Remove annoying torch.nn.Upsample warnings

    np.random.seed(123)
    t.manual_seed(123)
    random.seed(123)

    train_dataset, test_dataset = prepare_datasets(
        './data/raw/birdsong-recognition/train.csv',
        './data/processed/mel_specs_train/',
        123
    )

    print('Created datasets')

    print('Estimating data range')
    max_log = max([t.log(t.max(x['mel_spec']) + 0.0001) for x in train_dataset])
    min_log = min([t.log(t.min(x['mel_spec']) + 0.0001) for x in train_dataset])

    # max_log = 13.1293
    # min_log = -9.2103

    print('max_log: ', max_log)
    print('min_log: ', min_log)

    collate = Collate(400, min_log, max_log)

    train_dataloader = t.utils.data.DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate,
        num_workers=12
    )

    test_dataloader = t.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate,
        num_workers=8
    )

    model = SimpleCNN(264, 128, 400, 2e-4)

    trainer = pl.Trainer(
        # fast_dev_run=True,
        deterministic=True,
        gpus=1,
        accumulate_grad_batches=4,
        row_log_interval=64,
        auto_lr_find=True,
        callbacks=[pl.callbacks.LearningRateLogger()],
        early_stop_callback=pl.callbacks.EarlyStopping(
            monitor='top_1',
            min_delta=0.005,
            patience=5,
            mode='max'
        ),
        checkpoint_callback=pl.callbacks.ModelCheckpoint(
            monitor='top_1',
            save_last=True,
            save_top_k=3,
            mode='max'
        )
    )

    trainer.fit(model, train_dataloader, test_dataloader)


if __name__ == '__main__':
    main()
