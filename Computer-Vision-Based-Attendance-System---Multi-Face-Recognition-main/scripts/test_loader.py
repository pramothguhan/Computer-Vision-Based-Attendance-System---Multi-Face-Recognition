from src.data.dataset import CelebADataset
from src.data.transforms import get_train_transforms, get_val_transforms
from torch.utils.data import DataLoader

# Paths
root_dir = "data/celeba_custom_split"
train_split = os.path.join(root_dir, "train_identity.txt")
val_split = os.path.join(root_dir, "val_identity.txt")

# Load datasets
train_ds = CelebADataset(root_dir, train_split, transform=get_train_transforms())
val_ds = CelebADataset(root_dir, val_split, transform=get_val_transforms())

# Loaders
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

# Inspect
print(f"Train size: {len(train_ds)} | Classes: {train_ds.num_classes}")
print(f"Val size: {len(val_ds)}")

# Fetch one batch
images, labels = next(iter(train_loader))
print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)