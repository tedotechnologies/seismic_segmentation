import os
import numpy as np
import cv2
import random

try:
    import albumentations as A
except ImportError:
    A = None

def load_dat_file(filepath, shape=(224, 224)):
    data = np.fromfile(filepath, dtype=np.float32)
    return data.reshape(shape)

def load_cube(filepath, shape=(256, 256, 256), dtype=np.float32):
    data = np.fromfile(filepath, dtype=dtype)
    if data.size != np.prod(shape):
        raise ValueError(
            f"Размер данных {data.size} не совпадает с ожидаемой формой {shape} для файла {filepath}"
            )
    return data.reshape(shape)

def generate_point_prompt(mask):
    pos_indices = np.argwhere(mask > 0)
    if len(pos_indices) > 0:
        chosen_idx = random.choice(pos_indices)
        # SAM ожидает координаты в формате (x, y) = (колонка, строка)
        return np.array([chosen_idx[1], chosen_idx[0]], dtype=np.float32)
    return None

def apply_augmentations(sample, aug_pipeline):
    """
    Применяет аугментации к сэмплу.
    Ожидается, что aug_pipeline - это объект albumentations.Compose.
    Аугментация применяется к "seismic_img" и "label". 
    Если в сэмпле присутствует "mask_prompt", то и к нему тоже.
    """
    # Собираем словарь для аугментаций
    data = {"image": sample["seismic_img"], "mask": sample["label"]}
    # Если есть дополнительная маска-промпт, добавляем её (albumentations поддерживает несколько масок)
    if sample.get("mask_prompt") is not None:
        data["mask2"] = sample["mask_prompt"]

    augmented = aug_pipeline(**data)
    sample["seismic_img"] = augmented["image"]
    sample["label"] = augmented["mask"]
    if "mask2" in augmented:
        sample["mask_prompt"] = augmented["mask2"]

    return sample

# Универсальный датасет для сейсмических данных
class SeismicDataset:
    def __init__(self, config):
        """
        config: словарь с параметрами, например:
            {
                "type": "2D" или "3D" или "3D_variant",
                "seismic_dir": путь к сейсмическим данным,
                "label_dir": путь к меткам,
                "shape": (h, w) для 2D или (d, h, w) для 3D,
                "mask_dtype": np.uint8 или np.uint32,
                "num_slices": количество срезов для выборки (только для 3D),
                "neighbor_offset": шаг для выбора соседнего среза (для маски-промпта),
                "augmentation_pipeline": объект albumentations.Compose (опционально)
            }
        """
        self.data_type = config["type"]
        self.seismic_dir = config["seismic_dir"]
        self.label_dir = config["label_dir"]
        self.shape = config.get("shape", (224, 224))  # Исходная форма данных
        self.mask_dtype = config.get("mask_dtype", np.uint8)
        self.num_slices = config.get("num_slices", 1)  # используется для 3D
        self.neighbor_offset = config.get("neighbor_offset", 1)
        self.files = sorted(os.listdir(self.seismic_dir))
        # Целевая форма для унификации изображений и масок
        self.target_size = (224, 224)
        self.augmentation_pipeline = config.get("augmentation_pipeline", None)
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        filename = self.files[idx]
        seismic_path = os.path.join(self.seismic_dir, filename)
        label_path = os.path.join(self.label_dir, filename)
        
        sample = {"filename": filename}
        
        if self.data_type == "2D":
            # Загрузка 2D снимка и маски
            seismic = load_dat_file(seismic_path, shape=self.shape)
            label = load_dat_file(label_path, shape=self.shape).astype(self.mask_dtype)
            
            # Если сейсмика одноканальная, дублируем канал для создания 3-канального изображения
            if seismic.ndim == 2:
                seismic_img = np.stack([seismic] * 3, axis=-1)
            else:
                seismic_img = seismic
            
            # Приводим изображение и маску к размеру 224x224
            # seismic_img = cv2.resize(seismic_img, self.target_size, interpolation=cv2.INTER_LINEAR)
            # label = cv2.resize(label, self.target_size, interpolation=cv2.INTER_NEAREST)
            
            # Генерация точечного промпта из текущей маски
            point_prompt = generate_point_prompt(label)
            mask_prompt = None
            
            sample.update({
                "seismic_img": seismic_img,
                "label": label,
                "point_prompt": point_prompt,
                "mask_prompt": mask_prompt
            })
        
        elif self.data_type in ["3D", "3D_variant"]:
            # Загрузка 3D-куба сейсмики и меток
            seismic_cube = load_cube(seismic_path, shape=self.shape, dtype=np.float32)
            label_cube = load_cube(label_path, shape=self.shape, dtype=self.mask_dtype)
            depth = seismic_cube.shape[0]
            
            # Выбираем случайный срез
            slice_idx = random.choice(range(depth))
            seismic_slice = seismic_cube[slice_idx, :, :]
            label_slice = label_cube[slice_idx, :, :]
            
            # Создаём 3-канальное изображение из среза
            seismic_img = np.stack([seismic_slice] * 3, axis=-1)
            
            # Приводим срез и маску к размеру 224x224
            # seismic_img = cv2.resize(seismic_img, self.target_size, interpolation=cv2.INTER_LINEAR)
            # label_slice = cv2.resize(label_slice, self.target_size, interpolation=cv2.INTER_NEAREST)
            
            # Генерация точечного промпта для выбранного среза
            point_prompt = generate_point_prompt(label_slice)
            
            # Генерация маски-промпта: выбираем соседний срез с учетом neighbor_offset
            neighbor_idx = slice_idx + self.neighbor_offset
            if neighbor_idx < depth:
                mask_prompt = label_cube[neighbor_idx, :, :]
                mask_prompt = cv2.resize(mask_prompt, self.target_size, interpolation=cv2.INTER_NEAREST)
            else:
                mask_prompt = None
            
            sample.update({
                "seismic_img": seismic_img,
                "label": label_slice,
                "point_prompt": point_prompt,
                "mask_prompt": mask_prompt,
                "slice_idx": slice_idx
            })
        
        # Применение аугментаций, если они заданы в конфигурации
        if self.augmentation_pipeline is not None:
            sample = apply_augmentations(sample, self.augmentation_pipeline)
        
        return sample

# Функция для объединения нескольких источников данных
def create_combined_dataset(configs):
    """
    Принимает список конфигураций, создаёт датасет для каждой и объединяет данные.
    В реальной задаче можно использовать torch.utils.data.ConcatDataset или иной подход.
    """
    combined_samples = []
    for cfg in configs:
        dataset = SeismicDataset(cfg)
        for i in range(len(dataset)):
            combined_samples.append(dataset[i])
    return combined_samples

# Пример конфигураций для разных источников
if __name__ == "__main__":
    # Пример аугментационной пайплайна через Albumentations
    if A is not None:
        augmentation_pipeline = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5)
        ], additional_targets={"mask2": "mask"})
    else:
        augmentation_pipeline = None

    # Конфигурация для 2D данных
    config_2d = {
        "type": "2D",
        "seismic_dir": "/home/dmatveev/workdir/rosneft_segmentation/data/Salt2d/seismic",
        "label_dir": "/home/dmatveev/workdir/rosneft_segmentation/data/Salt2d/label",
        "shape": (224, 224),
        "mask_dtype": np.uint8,
        "augmentation_pipeline": augmentation_pipeline
    }
    
    # Конфигурация для 3D данных
    config_3d = {
        "type": "3D",
        "seismic_dir": "/home/dmatveev/workdir/rosneft_segmentation/data/sabamrine/seismic",
        "label_dir": "/home/dmatveev/workdir/rosneft_segmentation/data/sabamrine/label",
        "shape": (256, 256, 256),
        "mask_dtype": np.uint8,
        "num_slices": 3,
        "neighbor_offset": 1,
        "augmentation_pipeline": augmentation_pipeline
    }
    
    # Конфигурация для 3D данных с другим типом масок (например, paleokart)
    config_3d_variant = {
        "type": "3D_variant",
        "seismic_dir": "/home/dmatveev/workdir/rosneft_segmentation/data/paleokart/noise",
        "label_dir": "/home/dmatveev/workdir/rosneft_segmentation/data/paleokart/karst",
        "shape": (256, 256, 256),
        "mask_dtype": np.uint32,
        "num_slices": 3,
        "neighbor_offset": 1,
        "augmentation_pipeline": augmentation_pipeline
    }
    
    # Объединяем все источники в один датасет
    all_configs = [config_2d, config_3d, config_3d_variant]
    dataset = create_combined_dataset(all_configs)
    
    # Пример проверки: вывод информации о случайном сэмпле
    sample = random.choice(dataset)
    print("Файл:", sample.get("filename"))
    if "slice_idx" in sample:
        print("Срез:", sample["slice_idx"])
    print("Размер изображения:", sample["seismic_img"].shape)
    print("Размер маски:", sample["label"].shape)
