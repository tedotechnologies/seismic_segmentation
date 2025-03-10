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
from prepare_data import SegmentationDataset


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

def generate_input_points(label: np.ndarray) -> list:
    h, w = label.shape
    nonzero_indices = np.argwhere(label == 1)
    if len(nonzero_indices) > 0:
        idx = random.randint(0, len(nonzero_indices) - 1)
        y, x = nonzero_indices[idx]
        return [[x, y]]
    else:
        return [[w / 2.0, h / 2.0]]

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

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    os.environ["CLEARML_WEB_HOST"] = "https://app.clear.ml/"
    os.environ["CLEARML_API_HOST"] = "https://api.clear.ml"
    os.environ["CLEARML_FILES_HOST"] = "https://files.clear.ml"


    task = Task.init(project_name=cfg.clearml.project_name,
                     task_name=cfg.clearml.task_name)
    task.connect(OmegaConf.to_container(cfg, resolve=True))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Загрузка модели и processor
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

    # Преобразование строки mask_dtype в настоящий тип (если необходимо)
    if isinstance(cfg.data.mask_dtype, str):
        if cfg.data.mask_dtype == "np.uint8":
            mask_dtype = np.uint8
        elif cfg.data.mask_dtype == "np.uint32":
            mask_dtype = np.uint32
        else:
            mask_dtype = np.uint8
    else:
        mask_dtype = cfg.data.mask_dtype

    # Создание датасета
    dataset = SegmentationDataset({
        "type": cfg.data.get("type", "2D"),
        "seismic_dir": cfg.data.seismic_dir,
        "label_dir": cfg.data.label_dir,
        "shape": tuple(cfg.data.shape),
        "mask_dtype": mask_dtype,
        "use_pil": True
    })

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        collate_fn=custom_collate
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        collate_fn=custom_collate
    )

    for epoch in range(cfg.training.epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", ncols=100)
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()

            labels = torch.stack([ (torch.tensor(lbl) != 0).float() for lbl in batch["label"]]).to(device)
            batch_input_points = [generate_input_points(label) for label in labels.cpu().numpy()]
            inputs = processor(batch["seismic_img"], input_points=batch_input_points, return_tensors="pt")
            original_sizes = inputs["original_sizes"]
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            pred_logits = select_best_mask(outputs)

            if pred_logits.shape[-2:] != labels.shape[-2:]:
                pad_size = cfg.data.get("pad_size", {"height": 1024, "width": 1024})
                pred_logits = interpolate_prediction(pred_logits, original_sizes, pad_size, labels)

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


    print("Training complete.")

    test_data_cfg = {
        "type": "2D",
        "seismic_dir": "/home/dmatveev/workdir/rosneft_segmentation/data/paleokart/seismic",
        "label_dir": "/home/dmatveev/workdir/rosneft_segmentation/data/paleokart/label",
        "shape": (256, 256),
        "mask_dtype": np.uint8,
        "pad_size": {"height": 1024, "width": 1024}
    }
    full_test_dataset = SegmentationDataset(test_data_cfg)
    try:
        test_dataset = full_test_dataset[-200:]
    except TypeError:
        total = len(full_test_dataset)
        indices = list(range(total - 200, total))
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
            labels = torch.stack([(torch.tensor(lbl) != 0).float() for lbl in batch["label"]]).to(device)
            batch_input_points = [generate_input_points(label) for label in labels.cpu().numpy()]
            inputs_test = processor(batch["seismic_img"], input_points=batch_input_points, return_tensors="pt")
            original_sizes = inputs_test["original_sizes"]
            inputs_test = {k: v.to(device) for k, v in inputs_test.items()}
            outputs = model(**inputs_test)
            pred_logits = select_best_mask(outputs)

            if pred_logits.shape[-2:] != labels.shape[-2:]:
                pad_size = test_data_cfg.get("pad_size", {"height": 1024, "width": 1024})
                pred_logits = interpolate_prediction(pred_logits, original_sizes, pad_size, labels)

            test_loss = F.binary_cross_entropy_with_logits(pred_logits, labels.float())
            pred_masks = (torch.sigmoid(pred_logits) > 0.5).float().cpu().numpy()
            labels_np = labels.cpu().numpy()
            for pred_mask, label in zip(pred_masks, labels_np):
                iou, dice = compute_metrics(pred_mask, label)
                test_ious.append(iou)
                test_dices.append(dice)

    avg_test_iou = np.mean(test_ious) if test_ious else 0.0
    avg_test_dice = np.mean(test_dices) if test_dices else 0.0
    print(f"Test - Average IoU: {avg_test_iou:.4f}, Average Dice: {avg_test_dice:.4f}")
    task.get_logger().report_scalar("test_iou", "test", iteration=0, value=avg_test_iou)
    task.get_logger().report_scalar("test_dice", "test", iteration=0, value=avg_test_dice)


if __name__ == "__main__":
    main()
