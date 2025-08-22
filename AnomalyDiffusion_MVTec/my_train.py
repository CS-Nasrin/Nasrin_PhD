import sys
import os
import torch
import argparse
import pytorch_lightning as pl
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# === Inject command-line arguments ===
sys.argv += [
    "--actual_resume", "models/ldm/text2img-large/model.ckpt",
    "--sample_name", "triangle",
    "--anomaly_name", "scratch",
    "--spatial_encoder_embedding",
    "--base", "configs/latent-diffusion/txt2img-1p4B-finetune-encoder+embedding.yaml"
]

# === Define CLI arguments ===
parser = argparse.ArgumentParser()
parser.add_argument("--spatial_encoder_embedding", action="store_true", help="Enable spatial encoder embedding training")
opt, _ = parser.parse_known_args()

# === Set random seed ===
def setup_seed(seed):
    import random, numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# === Load model and config ===
def load_model(config_path, ckpt_path=None):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    if ckpt_path:
        print(f"Loading checkpoint from {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        model.load_state_dict(state_dict, strict=False)
    return model, config

# === Main training script ===
def main():
    print("🚀 Running Nasrin's clean training script!")
    setup_seed(42)

    config_path = "configs/latent-diffusion/txt2img-1p4B-finetune-encoder+embedding.yaml"
    #ckpt_path = "C:/my_new_desktop/anomalydiffusion-master/models/ldm/text2img-large/model.ckpt"
    ckpt_path = "C:\\my_new_desktop\\anomalydiffusion_train\\models\\ldm\\text2img-large\\model.ckpt"
    log_dir = "logs/my_train"

    model, config = load_model(config_path, ckpt_path)

    # ✅ Inject custom flag into the model
    model.spatial_encoder_embedding = opt.spatial_encoder_embedding
    model.learning_rate = config.model.base_learning_rate

    # ✅ Prepare the spatial encoder so it’s trainable
    if hasattr(model, "prepare_spatial_encoder"):
        model.prepare_spatial_encoder(optimze_together=True, data_enhance=True)

    datamodule = instantiate_from_config(config.data)

    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        precision=32,
        max_steps=config.lightning.trainer.max_steps,
        log_every_n_steps=10,
        detect_anomaly=True,
        callbacks=[
            ModelCheckpoint(dirpath=os.path.join(log_dir, "checkpoints"), every_n_train_steps=2),
        ],
        logger=TensorBoardLogger(save_dir=log_dir),
    )

    trainer.fit(model, datamodule=datamodule)
    # === Save spatial_encoder.pt separately ===
    #torch.save(model.embedding_manager.spatial_encoder_model.state_dict(), os.path.join(log_dir, "checkpoints", "spatial_encoder.pt"))


if __name__ == "__main__":
    main()
