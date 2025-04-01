import logging
import os
import random

import hydra
import numpy as np
import torch
from clearml import Task
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, IA3Config, TaskType, get_peft_model
from prepare_data import SegmentationDataset, create_combined_dataset
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import (
    SamModel,
    SamProcessor,
    get_cosine_schedule_with_warmup,
)

from utils import (
    augmentation_pipeline,
    combined_loss,
    compute_batch_metrics,
    custom_collate,
    parse_mask_dtype,
    postprocess_prediction,
    prepare_inputs,
    seed_everything,
    select_best_mask,
)

from settings import (
    CLEARML_WEB_HOST,
    CLEARML_API_HOST,
    CLEARML_FILES_HOST,
    CLEARML_API_ACCESS_KEY,
    CLEARML_API_SECRET_KEY,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    processor,
    device,
    cfg,
    task,
):

    model.train()
    epoch_loss = 0.0
    pbar = tqdm(train_loader, desc="Training", ncols=100)
    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()

        labels = torch.stack([(torch.tensor(lbl) != 0).float() for lbl in batch["label"]]).to(
            device
        )

        inputs, original_sizes, reshaped_input_sizes = prepare_inputs(batch, processor, device, cfg)
        outputs = model(**inputs, multimask_output=False)
        pred_logits = select_best_mask(outputs)
        pad_size = cfg.get("data", {}).get("pad_size", {"height": 1024, "width": 1024})
        pred_logits = postprocess_prediction(
            pred_logits, labels, original_sizes, reshaped_input_sizes, pad_size
        )
        # loss = F.binary_cross_entropy_with_logits(pred_logits, labels.float())
        loss = combined_loss(pred_logits, labels, alpha=0.3)
        loss.backward()
        optimizer.step()
        # scheduler.step()  # Update the learning rate scheduler
        epoch_loss += loss.item()

        if batch_idx % cfg.training.log_interval == 0:
            current_iter = batch_idx  # Simplified iteration count here; adjust as needed.
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}, LR: {current_lr}"
            )
            task.get_logger().report_scalar(
                "loss", "train", iteration=current_iter, value=loss.item()
            )

    avg_loss = epoch_loss / len(train_loader)
    logger.info(f"Average Training Loss: {avg_loss:.4f}")
    task.get_logger().report_scalar("epoch_loss", "train", iteration=0, value=avg_loss)
    return avg_loss


def validate(model, val_loader, processor, device, cfg, task):
    model.eval()
    ious, dices = [], []
    val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", ncols=100):
            labels = torch.stack([(torch.tensor(lbl) != 0).float() for lbl in batch["label"]]).to(
                device
            )
            inputs, original_sizes, reshaped_input_sizes = prepare_inputs(
                batch, processor, device, cfg
            )
            outputs = model(**inputs, multimask_output=False)
            pred_logits = select_best_mask(outputs)
            pad_size = cfg.get("data", {}).get("pad_size", {"height": 1024, "width": 1024})
            pred_logits = postprocess_prediction(
                pred_logits, labels, original_sizes, reshaped_input_sizes, pad_size
            )
            # batch_loss = F.binary_cross_entropy_with_logits(pred_logits, labels.float())
            batch_loss = combined_loss(pred_logits, labels.float(), alpha=0.3)
            val_loss += batch_loss.item()
            pred_masks = (torch.sigmoid(pred_logits) > 0.5).float().cpu().numpy()
            batch_ious, batch_dices = compute_batch_metrics(pred_masks, labels)
            ious.extend(batch_ious)
            dices.extend(batch_dices)
    avg_loss = val_loss / len(val_loader)
    avg_iou = np.mean(ious) if ious else 0.0
    avg_dice = np.mean(dices) if dices else 0.0
    logger.info(
        f"Validation - Loss: {avg_loss:.4f}, Average IoU: {avg_iou:.4f}, Average Dice: {avg_dice:.4f}"
    )
    task.get_logger().report_scalar("val_loss", "val", iteration=0, value=avg_loss)
    task.get_logger().report_scalar("val_iou", "val", iteration=0, value=avg_iou)
    task.get_logger().report_scalar("val_dice", "val", iteration=0, value=avg_dice)
    return avg_loss, avg_iou, avg_dice


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.training.seed)

    os.environ["CLEARML_WEB_HOST"] = CLEARML_WEB_HOST
    os.environ["CLEARML_API_HOST"] = CLEARML_API_HOST
    os.environ["CLEARML_FILES_HOST"] = CLEARML_FILES_HOST
    os.environ["CLEARML_API_ACCESS_KEY"] = CLEARML_API_ACCESS_KEY
    os.environ["CLEARML_API_SECRET_KEY"] = CLEARML_API_SECRET_KEY

    task = Task.init(project_name=cfg.clearml.project_name, task_name=cfg.clearml.task_name)
    task.connect(OmegaConf.to_container(cfg, resolve=True))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SamModel.from_pretrained(cfg.model.pretrained).to(device)
    processor = SamProcessor.from_pretrained(cfg.model.pretrained)

    # model = torch.compile(model, mode="default")

    if cfg.model.use_ia3:
        ia3_cfg = cfg.model.ia3_config
        target_modules = list(ia3_cfg.target_modules)
        feedforward_modules = list(ia3_cfg.feedforward_modules) if "feedforward_modules" in ia3_cfg else None
        ia3_config = IA3Config(
            task_type=TaskType.FEATURE_EXTRACTION,
            target_modules=target_modules,
            feedforward_modules=feedforward_modules,
            fan_in_fan_out=ia3_cfg.fan_in_fan_out,
            init_ia3_weights=ia3_cfg.init_ia3_weights,
        )
        model = get_peft_model(model, ia3_config)
    elif cfg.model.use_lora:
        lora_cfg = cfg.model.lora_config
        target_modules = list(lora_cfg.target_modules)
        lora_config = LoraConfig(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_cfg.lora_dropout,
            bias=lora_cfg.bias,
            task_type=TaskType.FEATURE_EXTRACTION,
            # modules_to_save=["mask_decoder", "vision_encoder"],
        )
        model = get_peft_model(model, lora_config)

    if cfg.model.freeze_base:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.mask_decoder.parameters():
            param.requires_grad = True
        for param in model.prompt_encoder.parameters():
            param.requires_grad = True

    logger.info(
        "Trainable parameters: %d", sum(p.numel() for p in model.parameters() if p.requires_grad)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr)

    train_dataset_configs = []
    for d in cfg.train_data:
        d_dict = dict(d)
        d_dict["mask_dtype"] = parse_mask_dtype(d_dict.get("mask_dtype", "np.uint8"))
        d_dict["shape"] = tuple(d_dict.get("shape", (224, 224)))
        d_dict["use_pil"] = True
        d_dict["augmentation_pipeline"] = augmentation_pipeline
        train_dataset_configs.append(d_dict)

    full_dataset = create_combined_dataset(train_dataset_configs)
    num_samples = len(full_dataset)
    indices = list(range(num_samples))

    random.shuffle(indices)
    train_split = int(cfg.training.train_val_split * num_samples)
    train_indices = indices[:train_split]
    val_indices = indices[train_split:]
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    if cfg.training.use_subset:
        subset_size = cfg.training.size_of_subset
        train_indices = list(range(min(subset_size, len(train_dataset))))
        train_dataset = Subset(train_dataset, train_indices)
        val_indices = list(range(min(subset_size, len(val_dataset))))
        val_dataset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        collate_fn=custom_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        collate_fn=custom_collate,
    )

    total_steps = cfg.training.epochs * len(train_loader)
    # scheduler = get_cosine_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=cfg.training.warmup_steps,
    #     num_training_steps=total_steps,
    # )
    scheduler=None

    for epoch in range(cfg.training.epochs):
        logger.info("Epoch %d/%d", epoch + 1, cfg.training.epochs)
        train_one_epoch(
            model, train_loader, optimizer, scheduler, processor, device, cfg, task
        )
        validate(model, val_loader, processor, device, cfg, task)

    logger.info("Training complete.")

    for _, test_cfg in enumerate(cfg.test_data):
        logger.info("Evaluating Test Dataset: %s", test_cfg.name)
        # Prepare test dataset config similar to training data
        test_data_cfg = {
            "type": test_cfg.get("type", "2D"),
            "seismic_dir": test_cfg.seismic_dir,
            "label_dir": test_cfg.label_dir,
            "shape": tuple(test_cfg.shape),
            "mask_dtype": np.uint8 if test_cfg.mask_dtype == "np.uint8" else test_cfg.mask_dtype,
            "pad_size": test_cfg.pad_size,
            "use_pil": True,
        }
        full_test_dataset = SegmentationDataset(test_data_cfg)

        test_loader = DataLoader(
            full_test_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            collate_fn=custom_collate,
        )

        model.eval()
        test_ious, test_dices = [], []
        with torch.no_grad():
            for batch in test_loader:
                labels = torch.stack([torch.tensor(lbl) for lbl in batch["label"]]).to(device)
                inputs, original_sizes, reshaped_input_sizes = prepare_inputs(
                    batch, processor, device, cfg
                )
                outputs = model(**inputs, multimask_output=False)
                pred_logits = select_best_mask(outputs)
                pad_size = test_data_cfg.get("pad_size", {"height": 1024, "width": 1024})
                pred_logits = postprocess_prediction(
                    pred_logits, labels, original_sizes, reshaped_input_sizes, pad_size
                )

                pred_masks = (torch.sigmoid(pred_logits) > 0.5).float().cpu().numpy()
                batch_ious, batch_dices = compute_batch_metrics(pred_masks, labels)
                test_ious.extend(batch_ious)
                test_dices.extend(batch_dices)

        avg_test_iou = np.mean(test_ious) if test_ious else 0.0
        avg_test_dice = np.mean(test_dices) if test_dices else 0.0
        logger.info(
            "Test Dataset: %s - Average IoU: %.4f, Average Dice: %.4f",
            test_cfg.name,
            avg_test_iou,
            avg_test_dice,
        )
        task.get_logger().report_scalar(
            "test_iou", f"test_dataset_{test_cfg.name}", iteration=0, value=avg_test_iou
        )
        task.get_logger().report_scalar(
            "test_dice", f"test_dataset_{test_cfg.name}", iteration=0, value=avg_test_dice
        )
        output_dir = cfg.training.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # if hasattr(model, "peft_config"):
        #     if isinstance(model.peft_config, dict):
        #         model.peft_config["target_modules"] = list(model.peft_config.get("target_modules", []))
        #     else:
        #         model.peft_config.target_modules = list(model.peft_config.target_modules)
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
