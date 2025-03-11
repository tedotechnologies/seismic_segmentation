import os
import random
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
from prepare_data import SegmentationDataset, create_combined_dataset

import albumentations as A

# Augmentation pipeline remains unchanged
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


def compute_metrics(
        pred: np.ndarray,
        target: np.ndarray,
        threshold: float = 0.5
        ) -> tuple:
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
    """
    Selects the best mask for each image based on the highest IoU.
    Supports both 5D (B, N, C, H, W) and 4D (B, N, H, W) outputs.
    Returns a tensor of shape (B, H, W).
    """
    if outputs.pred_masks.ndim == 5:
        # (B, N, C, H, W) assuming C==1
        pred_masks_candidates = outputs.pred_masks[:, 0, :, :, :]  # (B, N, H, W)
    else:
        # Assume (B, N, H, W)
        pred_masks_candidates = outputs.pred_masks[:, 0, :, :]
    
    iou_scores = outputs.iou_scores  # (B, N)
    best_masks = []
    for i in range(pred_masks_candidates.shape[0]):
        best_idx = torch.argmax(iou_scores[i])
        best_masks.append(pred_masks_candidates[i, best_idx, :, :])
    return torch.stack(best_masks, dim=0)  # (B, H, W)


def generate_input_points_and_labels(label: np.ndarray) -> tuple:
    """
    Returns a tuple (points, labels) where points is a list with at least one 2D point,
    and labels is a corresponding list. If no positive points exist, returns a dummy point ([0, 0])
    with a label -1.
    """
    h, w = label.shape
    nonzero_indices = np.argwhere(label == 1)
    if len(nonzero_indices) > 0:
        idx = random.randint(0, len(nonzero_indices) - 1)
        y, x = nonzero_indices[idx]
        return [[x, y]], [1]
    else:
        return [[0, 0]], [-1]


def interpolate_prediction(pred_logits: torch.Tensor,
                           original_sizes: list,
                           reshaped_input_sizes: list,
                           labels: torch.Tensor,
                           pad_size: dict) -> torch.Tensor:
    target_image_size = (pad_size["height"], pad_size["width"])

    pred_logits = pred_logits.unsqueeze(1)
    pred_logits = F.interpolate(
        pred_logits,
        size=target_image_size,
        mode="bilinear",
        align_corners=False
        )
    pred_logits = pred_logits.squeeze(1)
    
    pred_logits_list = []
    for i, rs in enumerate(reshaped_input_sizes):
        cropped = pred_logits[i, :rs[0], :rs[1]]
        orig_size = labels[i].shape
        upsampled = F.interpolate(cropped.unsqueeze(0).unsqueeze(0),
                                  size=orig_size, mode="bilinear", align_corners=False)
        pred_logits_list.append(upsampled.squeeze(0).squeeze(0))
    return torch.stack(pred_logits_list, dim=0)


def parse_mask_dtype(dtype_str):
    if isinstance(dtype_str, str):
        if dtype_str == "np.uint8":
            return np.uint8
        elif dtype_str == "np.uint32":
            return np.uint32
    return dtype_str


def prepare_inputs(batch: dict, processor, device: str) -> tuple:
    """
    Prepares inputs for the model from the batch.
    
    Returns:
      inputs: A dictionary of tensors on the specified device.
      original_sizes: Original image sizes (if provided by the processor).
      reshaped_input_sizes: Reshaped image sizes (if provided).
    """
    batch_input_points = []
    batch_input_labels = []
    for lbl in batch["label"]:
        label = torch.tensor(lbl)
        points, prompt_labels = generate_input_points_and_labels(label.cpu().numpy())
        batch_input_points.append(points)
        batch_input_labels.append(prompt_labels)

    inputs = processor(
        batch["seismic_img"],
        input_points=batch_input_points,
        input_labels=batch_input_labels,
        return_tensors="pt"
    )
    original_sizes = inputs.pop("original_sizes", None)
    reshaped_input_sizes = inputs.pop("reshaped_input_sizes", None)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs, original_sizes, reshaped_input_sizes


def postprocess_prediction(pred_logits: torch.Tensor,
                           labels: torch.Tensor,
                           original_sizes=None,
                           reshaped_input_sizes=None,
                           pad_size: dict = {"height": 1024, "width": 1024}) -> torch.Tensor:
    if pred_logits.shape[-2:] != labels.shape[-2:]:
        if original_sizes is not None and reshaped_input_sizes is not None:
            pred_logits = interpolate_prediction(pred_logits, original_sizes, reshaped_input_sizes, labels, pad_size)
        else:
            pred_logits = F.interpolate(pred_logits.unsqueeze(1),
                                        size=(pad_size["height"], pad_size["width"]),
                                        mode="bilinear", align_corners=False).squeeze(1)
    return pred_logits


def compute_batch_metrics(pred_masks, labels) -> tuple:
    ious, dices = [], []
    labels_np = labels.cpu().numpy()
    for pred_mask, label in zip(pred_masks, labels_np):
        iou, dice = compute_metrics(pred_mask, label)
        ious.append(iou)
        dices.append(dice)
    return ious, dices


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Set ClearML environment variables and initialize task
    os.environ["CLEARML_WEB_HOST"] = "https://app.clear.ml/"
    os.environ["CLEARML_API_HOST"] = "https://api.clear.ml"
    os.environ["CLEARML_FILES_HOST"] = "https://files.clear.ml"
    os.environ["CLEARML_API_ACCESS_KEY"] = "VH2OIPC5NKDGRNFJ9LW5W2KSU3YP4T"
    os.environ["CLEARML_API_SECRET_KEY"] = "Ixtz1NVs8wKDzNfyakkHIHHWN_Oy4vuzwbza8gu2za5SZpcl62e3s6v3s7uN9SzKbII"

    task = Task.init(project_name=cfg.clearml.project_name, task_name=cfg.clearml.task_name)
    task.connect(OmegaConf.to_container(cfg, resolve=True))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize model and processor
    model = SamModel.from_pretrained(cfg.model.pretrained).to(device)
    processor = SamProcessor.from_pretrained(cfg.model.pretrained)

    if cfg.model.freeze_base:
        for param in model.parameters():
            param.requires_grad = False

    if cfg.model.use_lora:
        lora_cfg = cfg.model.lora_config
        lora_config = LoraConfig(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.lora_alpha,
            target_modules=lora_cfg.target_modules,
            lora_dropout=lora_cfg.lora_dropout,
            bias=lora_cfg.bias,
            task_type=TaskType.FEATURE_EXTRACTION
        )
        model = get_peft_model(model, lora_config)

    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr)

    # Prepare training dataset configurations and create dataset
    train_dataset_configs = []
    for d in cfg.train_data:
        d_dict = dict(d)
        d_dict["mask_dtype"] = parse_mask_dtype(d_dict.get("mask_dtype", "np.uint8"))
        d_dict["shape"] = tuple(d_dict.get("shape", (224, 224)))
        d_dict["use_pil"] = True
        d_dict["augmentation_pipeline"] = augmentation_pipeline
        train_dataset_configs.append(d_dict)

    train_dataset = create_combined_dataset(train_dataset_configs)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # For faster iterations, take a subset for training and validation
    # indices = list(range(10))
    # train_subset = Subset(train_dataset, indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        collate_fn=custom_collate
    )
    
    # val_subset = Subset(val_dataset, indices)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        collate_fn=custom_collate
    )

    # Training loop
    for epoch in range(cfg.training.epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", ncols=100)
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()

            # Prepare labels for loss computation
            labels = torch.stack([(torch.tensor(lbl) != 0).float() for lbl in batch["label"]]).to(device)

            # Prepare inputs using the helper
            inputs, original_sizes, reshaped_input_sizes = prepare_inputs(batch, processor, device)
            outputs = model(**inputs)
            pred_logits = select_best_mask(outputs)
            # Safely obtain pad_size from config; defaults if key not present.
            pad_size = cfg.get("data", {}).get("pad_size", {"height": 1024, "width": 1024})
            pred_logits = postprocess_prediction(pred_logits, labels, original_sizes, reshaped_input_sizes, pad_size)
            
            loss = F.binary_cross_entropy_with_logits(pred_logits, labels.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if batch_idx % cfg.training.log_interval == 0:
                current_iter = epoch * len(train_loader) + batch_idx
                print(f"Epoch [{epoch+1}/{cfg.training.epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
                task.get_logger().report_scalar("loss", "train", iteration=current_iter, value=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{cfg.training.epochs}] Average Loss: {avg_loss:.4f}")
        task.get_logger().report_scalar("epoch_loss", "train", iteration=epoch, value=avg_loss)

        # Validation loop
        model.eval()
        ious, dices = [], []
        with torch.no_grad():
            for batch in val_loader:
                labels = torch.stack([torch.tensor(lbl) for lbl in batch["label"]]).to(device)
                inputs, original_sizes, reshaped_input_sizes = prepare_inputs(batch, processor, device)
                
                outputs = model(**inputs, multimask_output=False)
                pred_logits = select_best_mask(outputs)
                pad_size = cfg.get("data", {}).get("pad_size", {"height": 1024, "width": 1024})
                pred_logits = postprocess_prediction(pred_logits, labels, original_sizes, reshaped_input_sizes, pad_size)
                
                val_loss = F.binary_cross_entropy_with_logits(pred_logits, labels.float())
                
                pred_masks = (torch.sigmoid(pred_logits) > 0.5).float().cpu().numpy()
                batch_ious, batch_dices = compute_batch_metrics(pred_masks, labels)
                ious.extend(batch_ious)
                dices.extend(batch_dices)

        avg_iou = np.mean(ious) if ious else 0.0
        avg_dice = np.mean(dices) if dices else 0.0
        print(f"Validation - Loss: {val_loss.item():.4f}, Average IoU: {avg_iou:.4f}, Average Dice: {avg_dice:.4f}")
        task.get_logger().report_scalar("val_loss", "val", iteration=epoch, value=val_loss.item())
        task.get_logger().report_scalar("val_iou", "val", iteration=epoch, value=avg_iou)
        task.get_logger().report_scalar("val_dice", "val", iteration=epoch, value=avg_dice)

    print("Training complete.")

    # Testing
    test_data_cfg = {
        "type": cfg.test_data.get("type", "2D"),
        "seismic_dir": cfg.test_data.seismic_dir,
        "label_dir": cfg.test_data.label_dir,
        "shape": tuple(cfg.test_data.shape),
        "mask_dtype": np.uint8 if cfg.test_data.mask_dtype == "np.uint8" else cfg.test_data.mask_dtype,
        "pad_size": cfg.test_data.pad_size,
        "use_pil": True
    }
    full_test_dataset = SegmentationDataset(test_data_cfg)
    test_idx = 0

    try:
        test_dataset = full_test_dataset[-test_idx:]
    except TypeError:
        total = len(full_test_dataset)
        indices = list(range(total - test_idx, total))
        test_dataset = Subset(full_test_dataset, indices)

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        collate_fn=custom_collate
    )

    model.eval()
    test_ious, test_dices = [], []
    with torch.no_grad():
        for batch in test_loader:
            labels = torch.stack([torch.tensor(lbl) for lbl in batch["label"]]).to(device)
            inputs, original_sizes, reshaped_input_sizes = prepare_inputs(batch, processor, device)
            outputs = model(**inputs, multimask_output=False)
            pred_logits = select_best_mask(outputs)
            pad_size = test_data_cfg.get("pad_size", {"height": 1024, "width": 1024})
            pred_logits = postprocess_prediction(pred_logits, labels, original_sizes, reshaped_input_sizes, pad_size)
            
            pred_masks = (torch.sigmoid(pred_logits) > 0.5).float().cpu().numpy()
            batch_ious, batch_dices = compute_batch_metrics(pred_masks, labels)
            test_ious.extend(batch_ious)
            test_dices.extend(batch_dices)

    avg_test_iou = np.mean(test_ious) if test_ious else 0.0
    avg_test_dice = np.mean(test_dices) if test_dices else 0.0
    print(f"Test - Average IoU: {avg_test_iou:.4f}, Average Dice: {avg_test_dice:.4f}")
    task.get_logger().report_scalar("test_iou", "test", iteration=0, value=avg_test_iou)
    task.get_logger().report_scalar("test_dice", "test", iteration=0, value=avg_test_dice)

if __name__ == "__main__":
    main()
