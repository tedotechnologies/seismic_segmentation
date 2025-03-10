import os
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

from peft import get_peft_model, LoraConfig, TaskType
from transformers import SamModel, SamProcessor
from clearml import Task
from prepare_data import SegmentationDataset
from prepare_data import create_combined_dataset

import albumentations as A

augmentation_pipeline = A.Compose([
    A.Resize(height=256, width=256, p=1.0),
    # A.HorizontalFlip(p=0.5),
    # A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5)
], additional_targets={"mask2": "mask"})


def custom_collate(batch: list) -> dict:
    return {
        "filename": [item["filename"] for item in batch],
        "seismic_img": [item["seismic_img"] for item in batch],
        "label": [item["label"] for item in batch]
    }


def compute_metrics(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> tuple:
    pred_bin = (pred > threshold).astype(np.uint8)
    target_bin = (target > threshold).astype(np.uint8)
    if pred_bin.sum() == 0 and target_bin.sum() == 0:
        return 1.0, 1.0
    intersection = np.logical_and(pred_bin, target_bin).sum()
    union = np.logical_or(pred_bin, target_bin).sum()
    iou = intersection / union if union != 0 else 0.0
    pred_sum = pred_bin.sum()
    target_sum = target_bin.sum()
    dice = (2.0 * intersection) / (pred_sum + target_sum) if (pred_sum + target_sum) != 0 else 0.0
    return iou, dice


def select_best_mask(outputs) -> torch.Tensor:
    pred_masks_candidates = outputs.pred_masks[:, 0, :, :, :]
    iou_scores = outputs.iou_scores  # shape: [B, N]
    best_masks = []
    for i in range(pred_masks_candidates.shape[0]):
        best_idx = torch.argmax(iou_scores[i])
        best_masks.append(pred_masks_candidates[i, best_idx, :, :])
    return torch.stack(best_masks, dim=0)


def generate_input_points_and_labels(label: np.ndarray) -> tuple:
    """
    Returns a tuple (points, labels) where points is a list with at least one 2D point,
    and labels is a list with a corresponding label. If no positive point exists, returns
    a dummy point ([0, 0]) with label -1.
    """
    h, w = label.shape
    nonzero_indices = np.argwhere(label == 1)
    if len(nonzero_indices) > 0:
        idx = random.randint(0, len(nonzero_indices) - 1)
        y, x = nonzero_indices[idx]
        return [[x, y]], [1]  # valid prompt and label
    else:
        return [[0, 0]], [-1]   # dummy prompt and padding label


def interpolate_prediction(pred_logits: torch.Tensor, original_sizes: list, pad_size: dict, labels: torch.Tensor) -> torch.Tensor:
    target_image_size = (pad_size["height"], pad_size["width"])
    pred_logits = F.interpolate(pred_logits.unsqueeze(1), size=target_image_size, mode="bilinear", align_corners=False).squeeze(1)
    pred_logits_list = []
    for i, orig in enumerate(original_sizes):
        cropped = pred_logits[i, :orig[0], :orig[1]]
        orig_size = labels[i].shape
        upsampled = F.interpolate(cropped.unsqueeze(0).unsqueeze(0), size=orig_size, mode="bilinear", align_corners=False)
        pred_logits_list.append(upsampled.squeeze(0).squeeze(0))
    return torch.stack(pred_logits_list, dim=0)

def parse_mask_dtype(dtype_str):
    if isinstance(dtype_str, str):
        if dtype_str == "np.uint8":
            return np.uint8
        elif dtype_str == "np.uint32":
            return np.uint32
    return dtype_str

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    os.environ["CLEARML_WEB_HOST"] = "https://app.clear.ml/"
    os.environ["CLEARML_API_HOST"] = "https://api.clear.ml"
    os.environ["CLEARML_FILES_HOST"] = "https://files.clear.ml"
    os.environ["CLEARML_API_ACCESS_KEY"] = "VH2OIPC5NKDGRNFJ9LW5W2KSU3YP4T"
    os.environ["CLEARML_API_SECRET_KEY"] = "Ixtz1NVs8wKDzNfyakkHIHHWN_Oy4vuzwbza8gu2za5SZpcl62e3s6v3s7uN9SzKbII"

    task = Task.init(project_name=cfg.clearml.project_name,
                     task_name=cfg.clearml.task_name)
    task.connect(OmegaConf.to_container(cfg, resolve=True))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SamModel.from_pretrained(cfg.model.pretrained).to(device)
    processor = SamProcessor.from_pretrained(cfg.model.pretrained)

    if cfg.model.freeze_base:
        for param in model.parameters():
            param.requires_grad = False

from prepare_data import SeismicDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Train SAM model with ClearML")
    parser.add_argument("--project_name", type=str, default="SAM Fine Tuning")
    parser.add_argument("--task_name", type=str, default="LoRA Training")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze_base", action="store_true", help="Freeze base model parameters")
    return parser.parse_args()

def main():
    os.environ["CLEARML_WEB_HOST"] = "https://app.clear.ml/"
    os.environ["CLEARML_API_HOST"] = "https://api.clear.ml"
    os.environ["CLEARML_FILES_HOST"] = "https://files.clear.ml"
    os.environ["CLEARML_API_ACCESS_KEY"] = "VH2OIPC5NKDGRNFJ9LW5W2KSU3YP4T"
    os.environ["CLEARML_API_SECRET_KEY"] = "Ixtz1NVs8wKDzNfyakkHIHHWN_Oy4vuzwbza8gu2za5SZpcl62e3s6v3s7uN9SzKbII"

    args = parse_args()

    train_config = {
        "model": {
            "pretrained": "facebook/sam-vit-huge", 
            "use_lora": True,
            "lora_config": { 
                "r": 32,
                "lora_alpha": 64,
                "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj"],
                "lora_dropout": 0.1,
                "bias": "none",
                "task_type": "FEATURE_EXTRACTION"
            },
            "freeze_base": args.freeze_base
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "num_workers": 2,
            "log_interval": 10,
            "use_mask": True 
        },
        "data": {
            "type": "2D",
            "seismic_dir": "/home/dmatveev/workdir/rosneft_segmentation/data/Salt2d/seismic",
            "label_dir": "/home/dmatveev/workdir/rosneft_segmentation/data/Salt2d/label",
            "shape": (224, 224),
            "mask_dtype": np.uint8
        },
        "clearml": {
            "project_name": args.project_name,
            "task_name": args.task_name
        }
    }

    task = Task.init(project_name=train_config["clearml"]["project_name"],
                     task_name=train_config["clearml"]["task_name"])
    task.connect(train_config)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SamModel.from_pretrained(train_config["model"]["pretrained"]).to(device)
    processor = SamProcessor.from_pretrained(train_config["model"]["pretrained"])

    if train_config["model"]["freeze_base"]:
        for param in model.parameters():
            param.requires_grad = False

    if train_config["model"]["use_lora"]:
        lora_cfg = train_config["model"]["lora_config"]
        lora_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            target_modules=lora_cfg["target_modules"],
            lora_dropout=lora_cfg["lora_dropout"],
            bias=lora_cfg["bias"],
            task_type=TaskType.FEATURE_EXTRACTION
        )
        model = get_peft_model(model, lora_config)

    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config["training"]["lr"])

    data_config = {
        "seismic_dir": train_config["data"]["seismic_dir"],
        "label_dir": train_config["data"]["label_dir"],
        "shape": train_config["data"]["shape"],
    }
    seismic_dataset = SeismicDataset(data_config)
    torch_dataset = TorchSeismicDataset(seismic_dataset)
    train_loader = DataLoader(torch_dataset,
                              batch_size=train_config["training"]["batch_size"],
                              shuffle=True,
                              num_workers=train_config["training"]["num_workers"])

    num_epochs = train_config["training"]["epochs"]
    log_interval = train_config["training"]["log_interval"]

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", ncols=100)

        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()
            seismic_imgs = batch["seismic_img"].to(device)
            labels = batch["label"].to(device)

            inputs = processor(list(seismic_imgs), return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            image_embeddings = model.get_image_embeddings(inputs["pixel_values"])

            batch_prompts = []
            for label in labels.cpu().numpy():
                h, w = label.shape
                nonzero_indices = np.argwhere(label != 0)
                if len(nonzero_indices) > 0:
                    idx = random.randint(0, len(nonzero_indices) - 1)
                    y, x = nonzero_indices[idx]
                    prompt_point = [[x, y]]
                else:
                    prompt_point = [[w / 2.0, h / 2.0]]
                batch_prompts.append(prompt_point)

            prompt_inputs = processor(list(seismic_imgs), input_points=batch_prompts, return_tensors="pt")
            prompt_inputs = {k: v.to(device) for k, v in prompt_inputs.items()}
            prompt_inputs.pop("pixel_values", None)
            prompt_inputs.update({"image_embeddings": image_embeddings})

            outputs = model(**prompt_inputs)
            pred_masks = outputs.pred_masks

            if pred_masks.ndim == 5:
                pred_logits, _ = pred_masks.max(dim=2)  # [B, 1, H, W]
                pred_logits = pred_logits.squeeze(1)
            elif pred_masks.ndim == 4:
                pred_logits, _ = pred_masks.max(dim=1)  # [B, H, W]
            elif pred_masks.ndim == 3:
                pred_logits, _ = pred_masks.max(dim=0)  # [H, W]
                pred_logits = pred_logits.unsqueeze(0)  # [1, H, W]
            else:
                raise ValueError(f"Unexpected pred_masks shape: {pred_masks.shape}")

            if pred_logits.shape[-2:] != labels.shape[-2:]:
                pred_logits = F.interpolate(pred_logits.unsqueeze(1),
                                            size=labels.shape[-2:],
                                            mode="bilinear",
                                            align_corners=False).squeeze(1)

            loss = F.binary_cross_entropy_with_logits(pred_logits, labels.float())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if batch_idx % log_interval == 0:
                current_iter = epoch * len(train_loader) + batch_idx
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
                task.get_logger().report_scalar("loss", "train", iteration=current_iter, value=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")
        task.get_logger().report_scalar("epoch_loss", "train", iteration=epoch, value=avg_loss)

        checkpoint_path = f"checkpoint_epoch_{epoch+1}.pt"
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
        }, checkpoint_path)
        task.get_logger().report_artifact(name=f"checkpoint_epoch_{epoch+1}", artifact_object=checkpoint_path)

    print("Training complete.")

if __name__ == "__main__":
    main()
