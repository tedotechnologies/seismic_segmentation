import os
import random

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F

augmentation_pipeline = A.Compose(
    [
        A.Resize(height=256, width=256, p=1.0),
        # A.HorizontalFlip(p=0.5),
        # A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5)
    ],
    additional_targets={"mask2": "mask"},
)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def custom_collate(batch: list) -> dict:
    return {
        "filename": [item["filename"] for item in batch],
        "seismic_img": [item["seismic_img"] for item in batch],
        "label": [item["label"] for item in batch],
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
    """
    Select the best mask based on highest IoU.
    Supports outputs with shape (B, N, C, H, W) or (B, N, H, W).
    Returns a tensor of shape (B, H, W).
    """
    if outputs.pred_masks.ndim == 5:
        pred_masks_candidates = outputs.pred_masks[:, 0, :, :, :]  # (B, N, H, W)
    else:
        pred_masks_candidates = outputs.pred_masks[:, 0, :, :]

    iou_scores = outputs.iou_scores  # (B, N)
    best_masks = []
    for i in range(pred_masks_candidates.shape[0]):
        best_idx = torch.argmax(iou_scores[i])
        best_masks.append(pred_masks_candidates[i, best_idx, :, :])
    return torch.stack(best_masks, dim=0)


def generate_input_points_and_labels(label: np.ndarray, cfg=None) -> tuple:
    """
    Возвращает (points, labels) в зависимости от типа промпта.
    Для режима "point" выбираются случайные положительные точки в количестве,
    заданном cfg.prompt.num_points.
    Для режима "circle" генерируется указанное число кругов (cfg.prompt.num_circles).
      Для каждого круга сначала выбирается случайная положительная точка (центр),
      затем генерируются все точки внутри круга с диаметром, случайно выбранным
      вокруг cfg.prompt.circle_mean_diameter с допуском cfg.prompt.circle_diameter_delta.
    Для режима "bbox" делегирует вычисление функции generate_bbox_prompt.
    """
    h, w = label.shape
    positive_indices = np.argwhere(label == 1)
    prompt_type = cfg.get("prompt", {}).get("type", "point") if cfg is not None else "point"

    if prompt_type == "circle":
        num_circles = cfg.get("prompt", {}).get("num_circles", 1)
        points = []
        labels_list = []
        if len(positive_indices) == 0:
            return [[0, 0]], [-1]

        mean_diameter = cfg.get("prompt", {}).get("circle_mean_diameter", 10)
        delta = cfg.get("prompt", {}).get("circle_diameter_delta", 2)

        for _ in range(num_circles):
            idx = random.randint(0, len(positive_indices) - 1)
            y, x = positive_indices[idx]
            center_point = [int(x), int(y)]
            points.append(center_point)
            labels_list.append(1)

            diameter = random.uniform(mean_diameter - delta, mean_diameter + delta)
            radius = diameter / 2.0

            for j in range(max(0, int(y - radius)), min(h, int(y + radius) + 1)):
                for i in range(max(0, int(x - radius)), min(w, int(x + radius) + 1)):
                    if (i - x) ** 2 + (j - y) ** 2 <= radius ** 2:
                        # Пропускаем точку центра, чтобы не дублировать
                        if i == x and j == y:
                            continue
                        points.append([i, j])
                        labels_list.append(1)
        return points, labels_list

    elif prompt_type == "point":
        num_points = cfg.get("prompt", {}).get("num_points", 1)
        if len(positive_indices) > 0:
            if len(positive_indices) >= num_points:
                selected = positive_indices[
                    np.random.choice(len(positive_indices), num_points, replace=False)
                ]
            else:
                selected = positive_indices
            points = [[int(pt[1]), int(pt[0])] for pt in selected]
            return points, [1] * len(points)
        else:
            return [[0, 0]], [-1]

    elif prompt_type == "bbox":
        return generate_bbox_prompt(label, cfg), None


def interpolate_prediction(
    pred_logits: torch.Tensor,
    original_sizes: list,
    reshaped_input_sizes: list,
    labels: torch.Tensor,
    pad_size: dict,
) -> torch.Tensor:
    target_image_size = (pad_size["height"], pad_size["width"])
    pred_logits = pred_logits.unsqueeze(1)
    pred_logits = F.interpolate(
        pred_logits, size=target_image_size, mode="bilinear", align_corners=False
    )
    pred_logits = pred_logits.squeeze(1)

    pred_logits_list = []
    for i, rs in enumerate(reshaped_input_sizes):
        cropped = pred_logits[i, : rs[0], : rs[1]]
        orig_size = labels[i].shape
        upsampled = F.interpolate(
            cropped.unsqueeze(0).unsqueeze(0), size=orig_size, mode="bilinear", align_corners=False
        )
        pred_logits_list.append(upsampled.squeeze(0).squeeze(0))
    return torch.stack(pred_logits_list, dim=0)


def parse_mask_dtype(dtype_str):
    if isinstance(dtype_str, str):
        if dtype_str == "np.uint8":
            return np.uint8
        elif dtype_str == "np.uint32":
            return np.uint32
    return dtype_str


def generate_bbox_prompt(label: np.ndarray, cfg=None) -> list:
    """
    Computes a bounding box covering all positive pixels in label,
    adding a random error defined in cfg.
    Returns a list: [x_min, y_min, x_max, y_max].
    """
    h, w = label.shape
    nonzero_indices = np.argwhere(label == 1)
    if len(nonzero_indices) == 0:
        return [0, 0, 0, 0]

    y_coords, x_coords = nonzero_indices[:, 0], nonzero_indices[:, 1]
    x_min = int(x_coords.min())
    x_max = int(x_coords.max())
    y_min = int(y_coords.min())
    y_max = int(y_coords.max())

    bbox_error = cfg.get("prompt", {}).get("bbox_error", 5) if cfg is not None else 5

    x_min = max(0, x_min + random.randint(-bbox_error, bbox_error))
    y_min = max(0, y_min + random.randint(-bbox_error, bbox_error))
    x_max = min(w - 1, x_max + random.randint(-bbox_error, bbox_error))
    y_max = min(h - 1, y_max + random.randint(-bbox_error, bbox_error))

    return [x_min, y_min, x_max, y_max]


def pad_prompts(batch_points, batch_labels, pad_point=[0, 0], pad_label=-1):
    max_points = max(len(points) for points in batch_points)
    padded_points = []
    padded_labels = []
    for points, labels in zip(batch_points, batch_labels):
        n = len(points)
        if n < max_points:
            padded_points.append(points + [pad_point] * (max_points - n))
            padded_labels.append(labels + [pad_label] * (max_points - n))
        else:
            padded_points.append(points)
            padded_labels.append(labels)
    return padded_points, padded_labels


def generate_point_prompt(label: np.ndarray, cfg=None) -> tuple:
    """
    Генерирует промт в виде точек.
    Возвращает кортеж: (список точек, список меток).
    """
    positive_indices = np.argwhere(label == 1)
    num_points = cfg.get("prompt", {}).get("num_points", 1) if cfg else 1
    if len(positive_indices) > 0:
        if len(positive_indices) >= num_points:
            selected = positive_indices[
                np.random.choice(len(positive_indices), num_points, replace=False)
            ]
        else:
            selected = positive_indices
        points = [[int(pt[1]), int(pt[0])] for pt in selected]
        return points, [1] * len(points)
    else:
        return [[0, 0]], [-1]


def generate_circle_prompt(label: np.ndarray, cfg=None) -> tuple:
    """
    Генерирует промт в виде круга.
    Для каждого круга выбирается случайная положительная точка (центр),
    затем генерируются все точки внутри круга с диаметром, варьирующимся вокруг
    значения circle_mean_diameter с допуском circle_diameter_delta.
    """
    h, w = label.shape
    positive_indices = np.argwhere(label == 1)
    num_circles = cfg.get("prompt", {}).get("num_circles", 1) if cfg else 1
    points = []
    labels_list = []
    if len(positive_indices) == 0:
        return [[0, 0]], [-1]

    mean_diameter = cfg.get("prompt", {}).get("circle_mean_diameter", 10)
    delta = cfg.get("prompt", {}).get("circle_diameter_delta", 2)

    for _ in range(num_circles):
        idx = random.randint(0, len(positive_indices) - 1)
        y, x = positive_indices[idx]
        # Центр круга
        center_point = [int(x), int(y)]
        points.append(center_point)
        labels_list.append(1)

        diameter = random.uniform(mean_diameter - delta, mean_diameter + delta)
        radius = diameter / 2.0

        for j in range(max(0, int(y - radius)), min(h, int(y + radius) + 1)):
            for i in range(max(0, int(x - radius)), min(w, int(x + radius) + 1)):
                if (i - x) ** 2 + (j - y) ** 2 <= radius ** 2:
                    if i == x and j == y:
                        continue  # избегаем дублирования центра
                    points.append([i, j])
                    labels_list.append(1)
    return points, labels_list


def prepare_inputs(batch: dict, processor, device: str, cfg=None) -> tuple:
    """
    Подготавливает входные данные для модели с учётом типа промта.
    Добавлена поддержка комбинированных промтов:
      - "bbox+points": одновременно bounding box и точки
      - "bbox+circles": одновременно bounding box и круги
    """
    prompt_type = cfg.get("prompt", {}).get("type", "point") if cfg is not None else "point"

    if prompt_type == "bbox":
        batch_input_boxes = []
        for lbl in batch["label"]:
            label = torch.tensor(lbl)
            bbox = generate_bbox_prompt(label.cpu().numpy(), cfg)
            bbox = [float(coord) for coord in bbox]
            batch_input_boxes.append([bbox])
        inputs = processor(
            batch["seismic_img"],
            input_boxes=batch_input_boxes,
            return_tensors="pt",
        )
    elif prompt_type in ["bbox+points", "bbox+circles"]:
        batch_input_boxes = []
        batch_input_points = []
        batch_input_labels = []
        for lbl in batch["label"]:
            label = torch.tensor(lbl)
            # Генерация bounding box
            bbox = generate_bbox_prompt(label.cpu().numpy(), cfg)
            bbox = [float(coord) for coord in bbox]
            batch_input_boxes.append([bbox])
            # Генерация точечного или кругового промта
            if prompt_type == "bbox+points":
                points, prompt_labels = generate_point_prompt(label.cpu().numpy(), cfg)
            else:  # "bbox+circles"
                points, prompt_labels = generate_circle_prompt(label.cpu().numpy(), cfg)
            batch_input_points.append(points)
            batch_input_labels.append(prompt_labels)
        batch_input_points, batch_input_labels = pad_prompts(batch_input_points, batch_input_labels)
        inputs = processor(
            batch["seismic_img"],
            input_boxes=batch_input_boxes,
            input_points=batch_input_points,
            input_labels=batch_input_labels,
            return_tensors="pt",
        )
    else:
        # Для prompt типов "point" и "circle"
        batch_input_points = []
        batch_input_labels = []
        for lbl in batch["label"]:
            label = torch.tensor(lbl)
            if prompt_type == "circle":
                points, prompt_labels = generate_circle_prompt(label.cpu().numpy(), cfg)
            else:  # по умолчанию "point"
                points, prompt_labels = generate_point_prompt(label.cpu().numpy(), cfg)
            batch_input_points.append(points)
            batch_input_labels.append(prompt_labels)
        batch_input_points, batch_input_labels = pad_prompts(batch_input_points, batch_input_labels)
        inputs = processor(
            batch["seismic_img"],
            input_points=batch_input_points,
            input_labels=batch_input_labels,
            return_tensors="pt",
        )

    original_sizes = inputs.pop("original_sizes", None)
    reshaped_input_sizes = inputs.pop("reshaped_input_sizes", None)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs, original_sizes, reshaped_input_sizes


def postprocess_prediction(
    pred_logits: torch.Tensor,
    labels: torch.Tensor,
    original_sizes=None,
    reshaped_input_sizes=None,
    pad_size: dict = {"height": 1024, "width": 1024},
) -> torch.Tensor:
    if pred_logits.shape[-2:] != labels.shape[-2:]:
        if original_sizes is not None and reshaped_input_sizes is not None:
            pred_logits = interpolate_prediction(
                pred_logits, original_sizes, reshaped_input_sizes, labels, pad_size
            )
        else:
            pred_logits = F.interpolate(
                pred_logits.unsqueeze(1),
                size=(pad_size["height"], pad_size["width"]),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
    return pred_logits


def compute_batch_metrics(pred_masks, labels) -> tuple:
    ious, dices = [], []
    labels_np = labels.cpu().numpy()
    for pred_mask, label in zip(pred_masks, labels_np):
        iou, dice = compute_metrics(pred_mask, label)
        ious.append(iou)
        dices.append(dice)
    return ious, dices


def dice_loss(pred_logits: torch.Tensor, targets: torch.Tensor, epsilon=1e-7) -> torch.Tensor:
    """
    Computes the Dice loss.
    pred_logits: raw logits from the network.
    targets: ground truth binary masks.
    """
    pred_probs = torch.sigmoid(pred_logits)

    pred_flat = pred_probs.view(pred_probs.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    intersection = (pred_flat * targets_flat).sum(dim=1)
    denominator = (pred_flat**2).sum(dim=1) + (targets_flat**2).sum(dim=1)

    dice_score = (2 * intersection + epsilon) / (denominator + epsilon)
    loss = 1 - dice_score  # Dice loss
    return loss.mean()


def combined_loss(
    pred_logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.3
) -> torch.Tensor:
    """
    Computes the combined loss as a weighted sum of BCE loss and Dice loss.
    """
    bce_loss = F.binary_cross_entropy_with_logits(pred_logits, targets.float())
    d_loss = dice_loss(pred_logits, targets)
    return alpha * bce_loss + (1 - alpha) * d_loss
