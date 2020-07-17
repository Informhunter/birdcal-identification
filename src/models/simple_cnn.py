'''
Simple CNN on mel spectrograms with global avg/max pooling.
Trains on full audio length.
'''
import numpy as np
import torch as t
import pytorch_lightning as pl
from itertools import chain
from sklearn.metrics import roc_auc_score, average_precision_score
from src.data.dataset import EBIRD_CODE_TO_INDEX


class SimpleCNN(pl.LightningModule):
    '''
    Input: batch_size x n_mels x segment_size
    Output: batch_size x n_classes
    '''
    def __init__(self, n_classes, n_mels, segment_size, lr=0.001):
        super().__init__()
        self.conv_1 = t.nn.Conv2d(1, 64, (5, 5), padding=(2, 2))
        self.bn_1 = t.nn.BatchNorm2d(64)
        self.maxpool_1 = t.nn.MaxPool2d((4, 3))

        self.conv_2 = t.nn.Conv2d(64, 128, (5, 5), padding=(2, 2))
        self.bn_2 = t.nn.BatchNorm2d(128)
        self.maxpool_2 = t.nn.MaxPool2d((4, 3))

        self.conv_3 = t.nn.Conv2d(128, 256, (3, 3), padding=(1, 1))
        self.bn_3 = t.nn.BatchNorm2d(256)
        self.maxpool_3 = t.nn.MaxPool2d((4, 3))

        self.conv_4 = t.nn.Conv2d(256, 512, (3, 3), padding=(1, 1))
        self.maxpool_4 = t.nn.MaxPool2d((2, 3))
        self.bn_4 = t.nn.BatchNorm2d(512)

        # self.conv_a = t.nn.Conv1d(1, 8, 5, padding=1)
        # self.conv_b = t.nn.Conv1d(1, 8, 5)
        self.linear_y = t.nn.Linear(128, 128)

        self.dropout = t.nn.Dropout(0.5)
        # self.linear = t.nn.Linear(2816 + 128 * 2, n_classes)
        self.linear = t.nn.Linear(3328, n_classes)
        # self.linear_1 = t.nn.Linear(2816 + 128 * 2, 1024)
        # self.linear_2 = t.nn.Linear(1024, n_classes)

        self.n_mels = n_mels
        self.segment_size = segment_size
        self.n_classes = n_classes
        self.lr = lr
        # self.loss_fn = t.nn.BCEWithLogitsLoss()
        # self.loss_fn = t.nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def forward(self, batch):
        batch_size = len(batch['mel_specs'])
        n_segments = int(max(batch['segment_lengths']))
        mel_specs = batch['mel_specs']

        mel_specs = (
            mel_specs
            .transpose(-2, -1)
            .reshape(n_segments * batch_size, 1, -1, self.n_mels)
            .transpose(-2, -1)
        )

        # Mel features
        x = self.conv_1(mel_specs)
        x = self.bn_1(x)
        x = self.maxpool_1(x)
        x = t.relu(x)

        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.maxpool_2(x)
        x = t.relu(x)

        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.maxpool_3(x)
        x = t.relu(x)

        x = self.conv_4(x)
        x = self.bn_4(x)
        x = self.maxpool_4(x)

        x = x.view(batch_size, n_segments, -1)
        x = self.dropout(x)

        # Frequency features
        y = t.sum(mel_specs.view(batch_size, n_segments, self.n_mels, -1), dim=-1)
        # print('y.shape beginning: ', y.shape)
        y_max, _ = t.max(y, dim=-1)
        y_min, _ = t.min(y, dim=-1)
        y_max = y_max.unsqueeze(-1)
        y_min = y_min.unsqueeze(-1)

        # print('y_max.shape: ', y_max.shape)
        # print('y_min.shape: ', y_min.shape)

        y = (y - y_min) / (y_max - y_min + 0.0001)
        y = self.linear_y(y)

        # print('y.shape after linear: ', y.shape)

        # Global avg/max pooling
        segment_lengths = batch['segment_lengths'].view(-1, 1)

        for i in range(x.size(0)):
            x[i, int(segment_lengths[i].item()):, :] = 0
            y[i, int(segment_lengths[i].item()):, :] = 0

        x1 = t.sum(x, dim=1)
        x1 = x1 / segment_lengths
        x2, _ = t.max(x, dim=1)

        y1 = t.sum(y, dim=1)
        y1 = y1 / segment_lengths
        y2, _ = t.max(y, dim=1)

        # print('y1.shape: ', y1.shape)
        # print('y2.shape: ', y2.shape)

        z = t.cat((x1, x2, y1, y2), dim=-1)

        z = self.linear(z)

        # z = self.linear_1(z)
        # z = t.relu(z)
        # z = self.dropout(z)
        # z = self.linear_2(z)

        return z

    def training_step(self, batch, batch_idx):
        prediction = self(batch)
        loss = t.nn.functional.binary_cross_entropy_with_logits(
            prediction,
            batch['encoded_ebird_codes']
        )
        # loss = t.nn.functional.cross_entropy(
        #     prediction,
        #     t.argmax(batch['encoded_ebird_codes'], dim=-1)
        # )
        tensorboard_logs = {'train/loss': loss.item()}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        prediction = self(batch)
        loss = t.nn.functional.binary_cross_entropy_with_logits(
            prediction,
            batch['encoded_ebird_codes']
        )
        # loss = t.nn.functional.cross_entropy(
        #     prediction,
        #     t.argmax(batch['encoded_ebird_codes'], dim=-1)
        # )
        predicted_ranking = t.argsort(prediction, dim=1, descending=True)
        expected_indices = [EBIRD_CODE_TO_INDEX[x] for x in batch['primary_ebird_codes']]

        top_1 = []
        top_3 = []
        top_5 = []

        for expected_index, ranking in zip(expected_indices, predicted_ranking):
            if expected_index in ranking[:1]:
                top_1.append(1)
            else:
                top_1.append(0)

            if expected_index in ranking[:3]:
                top_3.append(1)
            else:
                top_3.append(0)

            if expected_index in ranking[:5]:
                top_5.append(1)
            else:
                top_5.append(0)
        return {
            'val_loss': loss,
            'top_1': top_1,
            'top_3': top_3,
            'top_5': top_5,
            'predictions': prediction.cpu().numpy(),
            'expectations': batch['encoded_ebird_codes'].cpu().numpy(),
        }

    def validation_epoch_end(self, outputs):
        val_loss_mean = t.stack([x['val_loss'] for x in outputs]).mean()
        top_1 = t.mean(t.FloatTensor(list(chain(*[x['top_1'] for x in outputs]))))
        top_3 = t.mean(t.FloatTensor(list(chain(*[x['top_3'] for x in outputs]))))
        top_5 = t.mean(t.FloatTensor(list(chain(*[x['top_5'] for x in outputs]))))

        predictions = np.concatenate([x['predictions'] for x in outputs])
        expectations = np.concatenate([x['expectations'] for x in outputs])

        try:
            roc_auc_macro = roc_auc_score(expectations, predictions, average='macro')
        except ValueError:
            roc_auc_macro = 0

        try:
            roc_auc_micro = roc_auc_score(expectations, predictions, average='micro')
        except ValueError:
            roc_auc_micro = 0

        try:
            roc_auc_samples = roc_auc_score(expectations, predictions, average='samples')
        except ValueError:
            roc_auc_samples = 0

        try:
            avg_pr_macro = average_precision_score(expectations, predictions, average='macro')
        except ValueError:
            avg_pr_macro = 0

        try:
            avg_pr_micro = average_precision_score(expectations, predictions, average='micro')
        except ValueError:
            avg_pr_micro = 0

        try:
            avg_pr_samples = average_precision_score(expectations, predictions, average='samples')
        except ValueError:
            avg_pr_samples = 0

        tensorboard_logs = {
            'val/loss': val_loss_mean,
            'val/top_1': top_1,
            'val/top_3': top_3,
            'val/top_5': top_5,
            'val/roc_auc_macro': roc_auc_macro,
            'val/roc_auc_micro': roc_auc_micro,
            'val/roc_auc_samples': roc_auc_samples,
            'val/avg_pr_macro': avg_pr_macro,
            'val/avg_pr_micro':  avg_pr_micro,
            'val/avg_pr_samples': avg_pr_samples,
        }

        return {
            'val_loss': val_loss_mean,
            'top_1': top_1,
            'top_3': top_3,
            'top_5': top_5,
            'val/roc_auc_macro': roc_auc_macro,
            'val/roc_auc_micro': roc_auc_micro,
            'val/roc_auc_samples': roc_auc_samples,
            'val/avg_pr_macro': avg_pr_macro,
            'val/avg_pr_micro':  avg_pr_micro,
            'val/avg_pr_samples': avg_pr_samples,
            'log': tensorboard_logs,
        }

    def configure_optimizers(self):
        optimizer = t.optim.Adam(self.parameters(), lr=self.lr)
        # scheduler = t.optim.lr_scheduler.CyclicLR(
        #     optimizer,
        #     self.lr / 4,
        #     self.lr * 2,
        #     step_size_up=500,
        #     cycle_momentum=False
        # )
        scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=4,
            threshold=0.0001,
            min_lr=1e-8,
        )
        return [optimizer], [{
            'scheduler': scheduler,
            'interval': 'step',
        }]


class Collate:
    def __init__(self, n_frames, min_log, max_log):
        self.min_log = min_log
        self.max_log = max_log
        self.n_frames = n_frames

    def __call__(self, batch):
        batch_size = len(batch)

        lengths = [x['mel_spec'].size(1) for x in batch]
        max_length = max(lengths)

        k = (max_length // self.n_frames) + 1
        padded_length = k * self.n_frames

        n_mels = batch[0]['mel_spec'].size(0)

        batched_mels = t.zeros(batch_size, n_mels, padded_length)

        segment_lengths = t.FloatTensor([(length // self.n_frames) + 1 for length in lengths])

        with t.no_grad():
            for i, item in enumerate(batch):
                mel_spec = t.log(item['mel_spec'] + 0.0001)
                mel_spec = (mel_spec - self.min_log) / (self.min_log - self.max_log)
                batched_mels[i, :, :mel_spec.size(1)] = mel_spec

        primary_labels = [x['primary_label'] for x in batch]
        secondary_labels = [x['secondary_labels'] for x in batch]
        durations = [x['duration'] for x in batch]

        encoded_ebird_codes = t.cat([x['encoded_ebird_codes'].view(1, -1) for x in batch], dim=0)
        primary_ebird_codes = [x['primary_ebird_code'] for x in batch]
        secondary_ebird_codes = list(chain(*[x['secondary_ebird_codes'] for x in batch]))

        return {
            'mel_specs': batched_mels,
            'original_lengths': lengths,
            'encoded_ebird_codes': encoded_ebird_codes,
            'primary_ebird_codes': primary_ebird_codes,
            'secondary_ebird_codes': secondary_ebird_codes,
            'primary_labels': primary_labels,
            'secondary_labels': secondary_labels,
            'segment_lengths': segment_lengths,
            'durations': durations,
        }
