import os, sys, argparse, json
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime

# Add src path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from data.dataset import CelebADataset
from data.transforms import get_transform_config
from models.simple_cnn import create_model, count_parameters


def accuracy_at_k(logits, targets, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = logits.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        return [(correct[:k].reshape(-1).float().sum(0).item() / targets.size(0)) * 100.0 for k in topk]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="data/celeba_custom_split")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--augment", type=str, choices=["basic", "standard", "aggressive"], default="standard")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"✅ Using device: {device}")

    # Transforms
    tf_config = get_transform_config(args.augment, (args.img_size, args.img_size))
    train_tf, val_tf = tf_config["train"], tf_config["val"]

    # Dataset file paths
    train_file = os.path.join(args.dataset_dir, "train_identity.txt")
    val_file = os.path.join(args.dataset_dir, "val_identity.txt")
    test_file = os.path.join(args.dataset_dir, "test_identity.txt")

    # Datasets & Dataloaders
    train_ds = CelebADataset(args.dataset_dir, train_file, transform=train_tf)
    val_ds = CelebADataset(args.dataset_dir, val_file, transform=val_tf)
    test_ds = CelebADataset(args.dataset_dir, test_file, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    # Model
    model = create_model(train_ds.num_classes).to(device)
    print(f"🎯 Model has {count_parameters(model):,} parameters across {train_ds.num_classes} classes.")

    # Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=2)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    os.makedirs("results", exist_ok=True)
    best_val_acc1 = 0.0

    # ========== TRAIN LOOP ==========
    for epoch in range(1, args.epochs + 1):
        print(f"\n🧪 Epoch {epoch}/{args.epochs}")
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = (correct / total) * 100.0

        # ========== VALIDATION ==========
        model.eval()
        val_loss, val_total = 0.0, 0
        val_acc1_sum, val_acc5_sum = 0.0, 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * labels.size(0)
                acc1, acc5 = accuracy_at_k(outputs, labels, topk=(1, 5))
                val_acc1_sum += acc1 * labels.size(0)
                val_acc5_sum += acc5 * labels.size(0)
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc1 = val_acc1_sum / val_total
        val_acc5 = val_acc5_sum / val_total
        scheduler.step(val_acc1)

        print(f"📊 Train Loss: {train_loss:.4f} | Acc@1: {train_acc:.2f}%")
        print(f"📈 Val   Loss: {val_loss:.4f} | Acc@1: {val_acc1:.2f}% | Acc@5: {val_acc5:.2f}%")

        # Save best model
        if val_acc1 > best_val_acc1:
            best_val_acc1 = val_acc1
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_classes": train_ds.num_classes,
                "epoch": epoch,
                "val_acc1": val_acc1,
            }, "results/best_model.pth")
            print(f"✅ Best model saved! acc@1: {val_acc1:.2f}%")

    # ========== TEST ==========
    print("\n🔍 Testing best model on test set...")
    ckpt = torch.load("results/best_model.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_loss, test_total = 0.0, 0
    test_acc1_sum, test_acc5_sum = 0.0, 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * labels.size(0)
            acc1, acc5 = accuracy_at_k(outputs, labels, topk=(1, 5))
            test_acc1_sum += acc1 * labels.size(0)
            test_acc5_sum += acc5 * labels.size(0)
            test_total += labels.size(0)

    test_loss /= test_total
    test_acc1 = test_acc1_sum / test_total
    test_acc5 = test_acc5_sum / test_total

    print(f"✅ Test Loss: {test_loss:.4f} | Acc@1: {test_acc1:.2f}% | Acc@5: {test_acc5:.2f}%")

    # Save results
    history = {
        "date": datetime.now().isoformat(),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "img_size": args.img_size,
        "best_val_acc1": best_val_acc1,
        "test_acc1": test_acc1,
        "test_acc5": test_acc5,
    }
    with open("results/training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\n🏁 Training complete. Results saved to `results/` folder.")


if __name__ == "__main__":
    main()
