import os
import torch

from collections import defaultdict
from torch import amp
from torch.utils.tensorboard import SummaryWriter
from src.console import track, Table
from src.optimizers import get_optimizer
from src.dataset import get_loader
from src.models import get_model
from src.losses import get_loss
from src.metrics import get_metric

class Trainer:
    def __init__(self, config):
        self.log_dir = os.path.join('logs', config.experiment, f'Fold_{config.fold}')
        self.train_loader, self.val_loader = get_loader(config)
        self.model = get_model(config).to(config.device)
        self.optimizer = get_optimizer(self.model, config)
        self.criterion = get_loss(config).to(config.device)
        self.metric_fn = get_metric(config).to(config.device)
        self.main_loss = config.loss.main_loss
        self.metric_name = config.metric.name
        self.device = config.device

        self.grad_scaler = amp.GradScaler(config.device)
        self.summary_writer = SummaryWriter(self.log_dir)
        self.columns = [*self.criterion.components, self.metric_name]
    def _run_epoch(self, train=True):
        if train:
            self.model.train()
            return self._run_loader(self.train_loader, 'Train', train=True)
        else:
            self.model.eval()
            return self._run_loader(self.val_loader, 'Val  ', train=False)
    def _run_loader(self, loader, desc, train=True):
        logs = defaultdict(float)

        grad_fn = torch.enable_grad if train else torch.inference_mode
        with grad_fn():
            for images, masks, _ in track(loader, desc=desc):
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)
                predicts, loss_dict = self._model_step(images, masks)
                loss = loss_dict[self.main_loss]

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.grad_scaler.scale(loss).backward()
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()

                batch_size = images.size(0)
                for name, loss in loss_dict.items():
                    logs[name] += loss.item() * batch_size
                self.metric_fn.update(predicts, masks)

        data_size = len(loader.dataset)
        logs = {
            name: loss / data_size
            for name, loss in logs.items()
        }
        logs[self.metric_name] = self.metric_fn.compute_reset()

        return logs
    def _model_step(self, images, masks):
        with torch.autocast(self.device):
            predicts = self.model(images)
            loss_dict = self.criterion(predicts, masks)
        return predicts, loss_dict
    def _write_summary(self, train_data, val_data, epoch):
        for title in self.columns:
            self.summary_writer.add_scalars(title, {'Train': train_data[title], 'Val': val_data[title]}, epoch)
    def fit(self, epochs):
        width = len(str(epochs))
        best_metric = 0
        for epoch in range(1, epochs + 1):
            print(f'{epoch:>{width}}/{epochs}')

            train_data = self._run_epoch(train=True)
            val_data = self._run_epoch(train=False)

            Table(
                ['', *self.columns],
                ['Train', *[train_data[column] for column in self.columns]],
                ['Val', *[val_data[column] for column in self.columns]]
            ).display()

            val_metric = val_data[self.metric_name]
            if val_metric >= best_metric:
                print(f'      Update {self.metric_name} {best_metric:.6f} -> {val_metric:.6f}')
                best_metric = val_metric
                torch.save(self.model.state_dict(), os.path.join(self.log_dir, 'best.pth'))

            self._write_summary(train_data, val_data, epoch)

        torch.save(self.model.state_dict(), os.path.join(self.log_dir, 'last.pth'))
        self.summary_writer.close()
