# 🎯 Computer Vision Attendance System — Multi-Face Recognition (CelebA)

This repository implements a deep learning pipeline to perform **identity recognition** using the **CelebA** dataset. It’s optimized to run on **HPC clusters** and avoids system crashes by skipping automatic unzipping of 200k+ images.

---

## 📦 1. Clone the Repository

```bash
git clone https://github.com/pramothguhan/Computer-Vision-Based-Attendance-System---Multi-Face-Recognition.git
cd Computer-Vision-Based-Attendance-System---Multi-Face-Recognition
```

Create a conda environment:

```bash
conda create -n celeba python=3.10 -y
conda activate celeba
pip install -r requirements.txt
```

---

## 📥 2. Download Annotations from Kaggle

This only fetches identity labels and partition info — **not images**.

```bash
python scripts/download.py --data-dir ./dataset
```

You’ll get:

```
dataset/
 ├─ identity_CelebA.txt
 ├─ list_eval_partition.txt
 └─ celeba-dataset.zip
```

---

## 🗂 3. Manually Unzip the Images (Important!)

To avoid system crashes during unzip, manually extract **only** the image folder from the ZIP:

```bash
mkdir -p data/celeba_custom_split/img_align_celeba
unzip -q dataset/celeba-dataset.zip 'img_align_celeba/*' -d data/celeba_custom_split/
```

Or, on some clusters:

```bash
7z x dataset/celeba-dataset.zip -o./data/celeba_custom_split/
```

After unzipping:

```bash
ls data/celeba_custom_split/img_align_celeba | head -n 5
000001.jpg
000002.jpg
...
```

---

## ✂️ 4. Generate Train / Val / Test Splits

This script generates `train_identity.txt`, `val_identity.txt`, and `test_identity.txt` from the annotation files.

```bash
python scripts/download_data.py \
  --data-dir ./data \
  --identity-file ./dataset/identity_CelebA.txt
```

---

## 🧪 5. Train the Model

```bash
python src/training/train.py \
  --dataset_dir data/celeba_custom_split \
  --batch_size 64 \
  --epochs 10 \
  --img_size 224 \
  --augment standard
```

You’ll see training progress and validation accuracy printed per epoch.

Model + logs will be saved to:

```
results/
 ├─ best_model.pth
 ├─ training_history.json
```

---

## 🧠 Want to Use a Different Model?

Add your new model architecture inside:

```
src/models/
```

Example:

```bash
src/models/custom_cnn.py
```

Then, update this line inside:

```python
# File: src/training/train.py
from models.simple_cnn import create_model
```

Change to:

```python
from models.custom_cnn import create_model
```

Make sure `create_model(num_classes)` is defined in your new model file.

---

## 🔁 Want to Train on a Subset of Images?

Open:

```python
# File: src/data/dataset.py
```

Locate inside `__init__`:

```python
if subset_ids:
    df = df[df.identity.isin(subset_ids)]
```

👉 You can pass your custom list of identity IDs (or even restrict the number of samples manually in that section) by modifying:

```python
df = df.head(5000)  # to use only first 5000 images (example)
```

Alternatively, update the `train_identity.txt` manually to include only selected identities or samples.

---

## 📁 Project Structure

```
src/
 ├─ data/
 │   ├─ dataset.py            # Custom Dataset
 │   ├─ transforms.py         # Augmentations
 ├─ models/
 │   └─ simple_cnn.py         # Default CNN model
 └─ training/
     └─ train.py              # Training script

scripts/
 ├─ download.py               # Download annotations
 ├─ download_data.py          # Split generator
results/
 └─ best_model.pth, logs, history.json
```

---

## 💡 Notes

- `.gitignore` already excludes `*.jpg`, `*.pth`, `*.log`, etc.
- Never extract 200k images using Python — unzip manually.
- All data is read from `data/celeba_custom_split/`.

---

## ✅ Final Checklist

- [x] Clone repo and create environment
- [x] Download annotation files only
- [x] **Manually unzip** image folder into `data/celeba_custom_split/`
- [x] Run `download_data.py` to create splits
- [x] Train using `train.py`

---

Made for clean runs in **resource-limited or HPC environments** 💪
