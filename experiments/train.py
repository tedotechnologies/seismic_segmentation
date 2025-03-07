import os
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from peft import get_peft_model, LoraConfig, TaskType
from transformers import SamModel, SamProcessor
from clearml import Task

from prepare_data import SegmentationDataset

def custom_collate(batch):
    return {
        "filename": [item[0] for item in batch],
        "seismic_img": [item[1] for item in batch],
        "label": [item[2] for item in batch]
    }

def parse_args():
    parser = argparse.ArgumentParser(description="Train SAM model with ClearML (batch inference using prompt points)")
    parser.add_argument("--project_name", type=str, default="SAM Fine Tuning")
    parser.add_argument("--task_name", type=str, default="LoRA Training")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze_base", action="store_true", help="Freeze base model parameters")
    return parser.parse_args()

def main():
    # Set ClearML environment variables
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
                "r": 32,  # adjusted to match notebook
                "lora_alpha": 32,
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
            "mask_dtype": np.uint8  # include mask dtype as in notebook
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

    data_cfg = train_config["data"]
    dataset = SegmentationDataset({
        "type": data_cfg.get("type", "2D"),
        "seismic_dir": data_cfg["seismic_dir"],
        "label_dir": data_cfg["label_dir"],
        "shape": data_cfg["shape"],
        "mask_dtype": data_cfg.get("mask_dtype", 0),
        "use_pil": True
    })

    train_loader = DataLoader(dataset,
                              batch_size=train_config["training"]["batch_size"],
                              shuffle=True,
                              num_workers=train_config["training"]["num_workers"],
                              collate_fn=custom_collate)

    num_epochs = train_config["training"]["epochs"]
    log_interval = train_config["training"]["log_interval"]

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", ncols=100)

        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()

            labels = torch.stack([torch.tensor(lbl) for lbl in batch["label"]]).to(device)

            batch_input_points = []
            for label in labels.cpu().numpy():
                h, w = label.shape
                nonzero_indices = np.argwhere(label == 1)
                if len(nonzero_indices) > 0:
                    idx = random.randint(0, len(nonzero_indices) - 1)
                    y, x = nonzero_indices[idx]
                    prompt_point = [[x, y]]
                else:
                    prompt_point = [[w / 2.0, h / 2.0]]
                batch_input_points.append(prompt_point)

            inputs = processor(batch["seismic_img"], input_points=batch_input_points, return_tensors="pt")
            original_sizes = inputs["original_sizes"]
            reshaped_input_sizes = inputs["reshaped_input_sizes"]
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)

            pred_logits = outputs.pred_masks[:, 0, 1, :, :]  # [B, H, W]

            if pred_logits.shape[-2:] != labels.shape[-2:]:
                # Шаг 1. Интерполируем до размера паддинга (целевого размера, к которому приводятся изображения)
                pad_size = train_config["data"].get("pad_size", {"height": 1024, "width": 1024})
                target_image_size = (pad_size["height"], pad_size["width"])
                pred_logits = F.interpolate(pred_logits.unsqueeze(1), size=target_image_size, mode="bilinear", align_corners=False).squeeze(1)
                
                # Шаг 2. Для каждого изображения из батча:
                pred_logits_list = []
                for i, rs in enumerate(reshaped_input_sizes):  # reshaped_input_sizes получаем из processor
                    # Обрезаем до размеров, на которые было изменено изображение перед паддингом
                    cropped = pred_logits[i, :rs[0], :rs[1]]
                    # Шаг 3. Интерполируем до оригинального размера меток
                    orig_size = labels[i].shape  # например, (224, 224)
                    upsampled = F.interpolate(cropped.unsqueeze(0).unsqueeze(0), size=orig_size, mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
                    pred_logits_list.append(upsampled)
                pred_logits = torch.stack(pred_logits_list, dim=0)

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
