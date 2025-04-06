# Rosneft Segmentation

This repository implements a segmentation approach for seismic images based on the methods described in the referenced article. The project fine-tunes the Segment Anything Model (SAM) for automatic segmentation of seismic features (e.g., paleovres, reefs) using an IA3 adapter and combined prompt strategies (bounding box + points). The approach has demonstrated improvements in segmentation metrics (Dice up to 0.60, IoU improvement) and processing speed (≈1.35 FPS), making it viable for industrial applications.

---

## 🔑 Key Features

### 🧠 Adapted SAM Model
Uses the pre-trained `facebook/sam-vit-huge` with an IA3 adapter (reducing trainable parameters by ~10%) to effectively segment seismic images.

### 🎯 Combined Prompt Strategy
Supports multiple prompt types including:
- Points  
- Circles  
- Bounding boxes  
- Combined (bounding box + points/circles)  
to enhance segmentation quality.

### 🛠️ Data Pipeline
Processes seismic datasets by:
- Converting 2D slices to RGB images  
- Normalizing data  
- Generating binary masks  

**Datasets include:**
- Real images: `Salt2D`
- Synthetic data: `sabamrine`, `paleokart`

### 🧪 Experiment Management
- Uses **Hydra** for flexible configuration  
- Integrates **ClearML** for automatic experiment tracking, reproducibility, and hyperparameter management

### 🚀 Training & Inference Pipeline
Implements:
- Training, validation, and testing pipelines  
- Automated metric reporting (IoU and Dice)

---

## 🧰 Installation

## 🧰 Installation

### Clone the Repository:
```bash
git clone <repository_url>
cd rosneft_segmentation
```

### Install Dependencies:
Make sure to activate a virtual environment beforehand.
```bash
pip install -r requirements.txt
```

### Download the SAM Model (optional):
Follow instructions from the [segment-anything](https://github.com/facebookresearch/segment-anything) repository.

---

## ⚙️ Configuration

### YAML Configurations
All experiment parameters are defined in `conf/config.yaml`.

Includes:

- **Model Settings:**  
  Pre-trained model, use of IA3 or LoRA adapter, freezing/unfreezing layers

- **Training Hyperparameters:**  
  Batch size, learning rate, number of epochs, logging intervals, etc.

- **Dataset Settings:**  
  Paths to seismic images and labels, image shapes, mask data types

- **Prompt Configurations:**  
  Prompt type (e.g., `"bbox+points"`), number of points/circles, bounding box error settings

---

## 🔐 ClearML Environment Variables

Before running experiments, update your ClearML credentials in the environment file:

Path: `experiments/.env.example`

```ini
CLEARML_WEB_HOST=https://app.clear.ml/
CLEARML_API_HOST=https://api.clear.ml
CLEARML_FILES_HOST=https://files.clear.ml
CLEARML_API_ACCESS_KEY=your_access_key
CLEARML_API_SECRET_KEY=your_secret_key
```

Rename the file to `.env` or set environment variables manually.

---

## 🏃 Running the Training

Execute the training script:

```bash
./run_train.sh
```

This calls `experiments/train.py` using Hydra-managed configurations. To run multiple experiments:

```bash
./run_train.sh --multirun
```

---

## 📁 Data Structure

```
/data
│
├── Salt2d       # Real 2D images of salt structures
├── sabamrine    # 3D cubes of synthetic paleovres
└── paleokart    # 3D synthetic images of paleokarts
```

---

## 🗂 Project Structure

```
conf/           # YAML configuration files (model, training, datasets, prompts)
experiments/    # Training scripts, utils, .env for ClearML
data/           # Seismic image data & labels
run_train.sh    # Shell script to initiate training
```

---

## 📌 Additional Information

- **Model:** `facebook/sam-vit-huge` with IA3 adapter  
- **Management:** Hydra + ClearML  
- **Usage:** Modify configs to fit your datasets and experiments  

---

For more details or assistance, please refer to the documentation in the repository or contact the project maintainer.