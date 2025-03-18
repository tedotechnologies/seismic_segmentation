import logging
import os
import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

try:
    import albumentations as A
except ImportError:
    A = None

logging.basicConfig(level=logging.INFO)


def load_dat_file(
    filepath: str,
    shape=(224, 224),
) -> np.ndarray:
    data = np.fromfile(filepath, dtype=np.float32)
    return data.reshape(shape)


def load_cube(
    filepath: str,
    shape=(256, 256, 256),
    dtype=np.float32,
) -> np.ndarray:

    data = np.fromfile(filepath, dtype=dtype)
    if data.size != np.prod(shape):
        raise ValueError(
            f"Размер данных {data.size} не совпадает с ожидаемой формой {shape} для файла {filepath}",
        )
    return data.reshape(shape)


def generate_point_prompt(mask: np.ndarray) -> np.ndarray:
    pos_indices = np.argwhere(mask > 0)
    if len(pos_indices) > 0:
        chosen_idx = random.choice(pos_indices)
        return np.array([chosen_idx[1], chosen_idx[0]], dtype=np.float32)
    return None


def apply_augmentations(sample, aug_pipeline):
    """
    Применяет аугментации к сэмплу.
    Ожидается, что aug_pipeline - это объект albumentations.Compose.
    Аугментация применяется к "seismic_img" и "label".
    Если в сэмпле присутствует "mask_prompt", то и к нему тоже.
    """
    data = {"image": sample["seismic_img"], "mask": sample["label"]}
    if sample.get("mask_prompt") is not None:
        data["mask2"] = sample["mask_prompt"]

    augmented = aug_pipeline(**data)
    sample["seismic_img"] = augmented["image"]
    sample["label"] = augmented["mask"]
    if "mask2" in augmented:
        sample["mask_prompt"] = augmented["mask2"]

    return sample


def to_pil_image(np_img: np.ndarray) -> Image.Image:
    """
    Преобразует numpy-массив в PIL Image. Если данные не uint8, выполняется нормализация.
    """
    if np_img.dtype != np.uint8:
        if np_img.max() != np_img.min():
            np_img = (255 * (np_img - np_img.min()) / (np_img.max() - np_img.min())).astype(
                np.uint8
            )
        else:
            np_img = np_img.astype(np.uint8)
    return Image.fromarray(np_img)


class SegmentationDataset(Dataset):
    """
    Датасет для сегментации. Принимает конфигурацию с параметрами:
      - type: тип данных ("2D", "3D" и т.п.)
      - seismic_dir: директория с сейсмическими данными
      - label_dir: директория с метками
      - shape: ожидаемая форма данных
      - mask_dtype: тип данных для меток
      - use_pil: если True, возвращает изображение в формате PIL, иначе numpy-массив
      - augmentation_pipeline: пайплайн аугментаций (albumentations.Compose)
    """

    def __init__(self, cfg: dict):
        self.data_type = cfg.get("type", "2D")
        self.seismic_dir = cfg["seismic_dir"]
        self.label_dir = cfg["label_dir"]
        self.shape = cfg.get("shape", (224, 224))
        self.mask_dtype = cfg.get("mask_dtype", 0)
        self.use_pil = cfg.get("use_pil", True)
        self.file_list = sorted(os.listdir(self.seismic_dir))
        self.augmentation_pipeline = cfg.get("augmentation_pipeline", None)

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        filename = self.file_list[idx]
        seismic_path = os.path.join(self.seismic_dir, filename)
        label_path = os.path.join(self.label_dir, filename)

        seismic = load_dat_file(seismic_path, shape=self.shape)
        label = load_dat_file(label_path, shape=self.shape)

        if seismic.ndim == 2:
            seismic_img = np.stack([seismic] * 3, axis=-1)
        else:
            seismic_img = seismic

        sample = {"filename": filename, "seismic_img": seismic_img, "label": label}

        if self.augmentation_pipeline is not None:
            sample = apply_augmentations(sample, self.augmentation_pipeline)

        if self.use_pil:
            sample["seismic_img"] = to_pil_image(sample["seismic_img"]).convert("RGB")

        return sample


def create_combined_dataset(configs: list) -> list:
    """
    Создаёт объединённый датасет из нескольких конфигураций.
    """
    combined_samples = []
    for cfg in configs:
        dataset = SegmentationDataset(cfg)
        combined_samples.extend([dataset[i] for i in range(len(dataset))])
    return combined_samples


if __name__ == "__main__":
    # Пример использования
    if A is not None:
        augmentation_pipeline = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            ],
            additional_targets={"mask2": "mask"},
        )
    else:
        augmentation_pipeline = None

    # Конфигурация для 2D данных
    config_2d = {
        "type": "2D",
        "seismic_dir": "/home/dmatveev/workdir/rosneft_segmentation/data/Salt2d/seismic",
        "label_dir": "/home/dmatveev/workdir/rosneft_segmentation/data/Salt2d/label",
        "shape": (224, 224),
        "mask_dtype": np.uint8,
        "augmentation_pipeline": augmentation_pipeline,
        "use_pil": True,
    }

    # Пример конфигураций для 3D (будущая поддержка)
    config_3d = {
        "type": "3D",
        "seismic_dir": "/home/dmatveev/workdir/rosneft_segmentation/data/sabamrine/seismic",
        "label_dir": "/home/dmatveev/workdir/rosneft_segmentation/data/sabamrine/label",
        "shape": (256, 256, 256),
        "mask_dtype": np.uint8,
        "num_slices": 3,
        "neighbor_offset": 1,
        "augmentation_pipeline": augmentation_pipeline,
    }

    config_3d_variant = {
        "type": "3D_variant",
        "seismic_dir": "/home/dmatveev/workdir/rosneft_segmentation/data/paleokart/noise",
        "label_dir": "/home/dmatveev/workdir/rosneft_segmentation/data/paleokart/karst",
        "shape": (256, 256, 256),
        "mask_dtype": np.uint32,
        "num_slices": 3,
        "neighbor_offset": 1,
        "augmentation_pipeline": augmentation_pipeline,
    }

    all_configs = [config_2d, config_3d, config_3d_variant]
    dataset = create_combined_dataset(all_configs)
    sample = random.choice(dataset)
    logging.info(f"Файл: {sample.get('filename')}")
    if isinstance(sample.get("seismic_img"), Image.Image):
        logging.info(f"Размер изображения: {sample['seismic_img'].size}")
    else:
        logging.info(f"Размер изображения: {sample['seismic_img'].shape}")
    logging.info(f"Размер маски: {sample['label'].shape}")
