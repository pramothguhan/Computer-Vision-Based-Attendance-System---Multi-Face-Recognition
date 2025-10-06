import argparse
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default="data/celeba_custom_split")
ap.add_argument("--partition-csv", default=None)
args = ap.parse_args()

data_dir = Path(args.data_dir)
part_csv = Path(args.partition_csv) if args.partition_csv else data_dir/"list_eval_partition.csv"
if not part_csv.exists(): raise FileNotFoundError(part_csv)

df = pd.read_csv(part_csv)
cols = [c.lower() for c in df.columns]; df.columns = cols
fname = "image_id" if "image_id" in cols else ("image" if "image" in cols else "filename")
out = df[[fname, "partition"]].copy(); out.columns = ["filename","partition"]
out.to_csv(data_dir/"list_eval_partition.txt", sep=" ", header=False, index=False)
print("✓ Wrote list_eval_partition.txt")