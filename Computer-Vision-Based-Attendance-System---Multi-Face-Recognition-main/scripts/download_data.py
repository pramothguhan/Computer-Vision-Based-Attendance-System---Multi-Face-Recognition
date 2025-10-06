import argparse, logging
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('dataset_setup.log'), logging.StreamHandler()])
log = logging.getLogger(__name__)

def main():
    ap = argparse.ArgumentParser(description="Split CelebA using only identity_CelebA.txt")
    ap.add_argument("--data-dir", default="data/celeba_custom_split")
    ap.add_argument("--identity-file", default="./data/identity_CelebA.txt")
    ap.add_argument("--train-size", type=float, default=0.8)
    ap.add_argument("--val-size", type=float, default=0.1)
    ap.add_argument("--random-seed", type=int, default=42)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = data_dir / "celeba_custom_split"
    out_dir.mkdir(parents=True, exist_ok=True)

    id_df = pd.read_csv(args.identity_file, sep=r"\s+", header=None, names=["filename", "identity"])

    # First split: Train and temp (val+test) - no stratify
    train_df, temp_df = train_test_split(
        id_df, train_size=args.train_size, random_state=args.random_seed
    )

    # Second split: Val and Test from temp - no stratify
    val_rel = args.val_size / (1 - args.train_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=1 - val_rel, random_state=args.random_seed
    )

    def dump(df, name):
        f = out_dir / f"{name}_identity.txt"
        df.to_csv(f, sep=" ", header=False, index=False)
        log.info(f"✓ Wrote {name} set with {len(df)} samples to {f}")
        return len(df)

    ntr = dump(train_df, "train")
    nv = dump(val_df, "val")
    nt = dump(test_df, "test")

    log.info("=" * 60)
    log.info(f"READY: {out_dir}")
    log.info(f"Train: {ntr} | Val: {nv} | Test: {nt}")
    log.info("=" * 60)

if __name__ == "__main__":
    main()