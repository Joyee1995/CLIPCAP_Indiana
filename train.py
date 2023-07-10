import os
import copy
import wandb
import torch
from pathlib import Path
from config import ex, config
import pytorch_lightning as pl
from datetime import datetime
from Dataset import ClipCocoDataset
from torch.utils.data import DataLoader
from Model import ClipCaptionPrefix, ClipCaptionModel

torch.manual_seed(0)

@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    with open(_config['wandb_api_key_fp'], 'r') as f:
        wandb_api_key = f.read().strip()
        wandb.login(key=wandb_api_key)
    datetime_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_version = _config['version'] if _config['version'] is not None else datetime_str
    _config['models_dir'] = os.path.join(_config["model_dir"], _config["name"], model_version)
    Path(_config['models_dir']).mkdir(parents=True, exist_ok=True)

    pl.seed_everything(_config["seed"])
    dataset_train = ClipCocoDataset(_config, split='train')
    dataset_valid = ClipCocoDataset(_config, split='valid')
    dataloader_train = DataLoader(dataset_train, batch_size=_config['bs'], shuffle=True, num_workers=_config['num_workers'])
    dataloader_valid = DataLoader(dataset_valid, batch_size=_config['bs'], shuffle=False, num_workers=_config['num_workers'])

    _config['num_training_steps'] = _config['epochs'] * len(dataloader_train)
    model = ClipCaptionPrefix(_config) if _config['only_prefix'] else ClipCaptionModel(_config)
    print("length of train dataset", len(dataset_train))
    print("length of valid dataset", len(dataset_valid))
    print("train on only prefix" if _config['only_prefix'] else "train on prefix + gpt")

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=_config['models_dir'], 
        filename='{epoch}-{valid_loss:.3f}',
        verbose=True,
        save_top_k=_config['save_top_k'], 
        every_n_epochs=1,
        monitor="valid_loss", 
        mode="min", 
        save_last=True)
    summary_callback = pl.callbacks.ModelSummary(max_depth=1)
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")

    csv_logger = pl.loggers.CSVLogger(save_dir=_config["log_dir"], name=_config['name'], version=datetime_str)
    csv_logger.log_hyperparams(_config)
    wandb_logger = pl.loggers.WandbLogger(project='MedicalReportGeneration', save_dir=_config["log_dir"], name=_config['name'], version=model_version)
    wandb_logger.experiment.config.update(_config, allow_val_change=True)
    trainer = pl.Trainer(max_epochs=_config['epochs'], 
                        logger=[csv_logger, wandb_logger], 
                        log_every_n_steps=(len(dataset_train) // _config['bs']) // 3,
                        callbacks=[checkpoint_callback, lr_callback, summary_callback])
    trainer.fit(model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_valid, ckpt_path=_config['ckpt_path'])

    # optimizer = model.configure_optimizers()
    # for batch_idx, (video_tensor, labels_onehot) in enumerate(val_loader):
    #     video_tensor, labels_onehot = video_tensor.to(device), labels_onehot.to(device)
    #     loss = model.training_step((video_tensor, labels_onehot), batch_idx)
    #     loss.backward()
    #     optimizer.step()
    #     optimizer.zero_grad()