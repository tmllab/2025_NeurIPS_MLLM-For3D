import os
import argparse
import torch
import torch.nn as nn
import MinkowskiEngine as ME
import pytorch_lightning as pl
from utils.read_config import generate_config
from pretrain.model_builder_lisa import make_model
from pytorch_lightning.strategies import DDPStrategy
from pretrain.lightning_trainer import LightningPretrain
from pretrain.lightning_datamodule_lisa import PretrainDataModule
from pretrain.lightning_trainer_spconv import LightningPretrainSpconv
from pytorch_lightning.loggers import TensorBoardLogger

def main():
    """
    Code for launching the pretraining
    """
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg_file", type=str, default="config/slidr_minkunet.yaml", help="specify the config for training"
    )
    parser.add_argument(
        "--resume_path", type=str, default=None, help="provide a path to resume an incomplete training"
    )
    args = parser.parse_args()
    config = generate_config(args.cfg_file)
    if args.resume_path:
        config['resume_path'] = args.resume_path

    if os.environ.get("LOCAL_RANK", 0) == 0:
        print(
            "\n" + "\n".join(list(map(lambda x: f"{x[0]:20}: {x[1]}", config.items())))
        )

    # Initialize Data Module
    dm = PretrainDataModule(config)

    # Initialize Models
    model_points, model_images, model_fusion = make_model(config)

    # Convert BatchNorm for multi-GPU training
    if config["num_gpus"] > 1:
        model_points = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model_points)
        model_images = nn.SyncBatchNorm.convert_sync_batchnorm(model_images)
        model_fusion = nn.SyncBatchNorm.convert_sync_batchnorm(model_fusion)

    # Choose the appropriate Lightning Module
    if config["model_points"] == "minkunet":
        module = LightningPretrain(model_points, model_images, model_fusion, config)
    elif config["model_points"] == "voxelnet":
        module = LightningPretrainSpconv(model_points, model_images, config)

    # Set up the working directory and logger
    path = os.path.join(config["working_dir"], config["datetime"])

    # Adding TensorBoard Logger for visualization
    logger = TensorBoardLogger(save_dir='/home/jiaxin.huang/jiaxin/reasonseg/tensorboard', name="tensorboard_logs")

    # Initialize the Trainer with the logger
    trainer = pl.Trainer(
        devices=config["num_gpus"],
        accelerator='gpu',
        default_root_dir=path,
        max_epochs=config["num_epochs"],
        strategy=DDPStrategy(find_unused_parameters=True),
        num_sanity_val_steps=0,
        check_val_every_n_epoch=10,
        logger=logger,  # Add TensorBoard logger here
        log_every_n_steps=10  # Frequency of logging metrics
    )

    print("Starting the training")
    trainer.fit(module, dm)


if __name__ == "__main__":
    main()
