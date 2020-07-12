import torch as t


class BCELossTrainingMixin:
    def training_step(self, batch, batch_idx):
        prediction = self(batch)
        loss = t.nn.functional.binary_cross_entropy_with_logits(
            prediction,
            batch['encoded_ebird_codes']
        )
        tensorboard_logs = {'train/loss': loss.item()}
        return {'loss': loss, 'log': tensorboard_logs}


class BCELossValidationMixin:
    def training_step(self, batch, batch_idx):
        prediction = self(batch)
        loss = t.nn.functional.binary_cross_entropy_with_logits(
            prediction,
            batch['encoded_ebird_codes']
        )
        tensorboard_logs = {'train/loss': loss.item()}
        return {'loss': loss, 'log': tensorboard_logs}
