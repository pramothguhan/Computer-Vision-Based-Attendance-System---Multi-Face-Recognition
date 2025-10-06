import os, argparse, logging, zipfile, subprocess, shutil
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler('kaggle_fetch.log'), logging.StreamHandler()])
log = logging.getLogger(__name__)

KAGGLE_DS = "jessicali9530/celeba-dataset"

def ensure_kaggle():
    try:
        import kaggle  # noqa
        log.info("✓ Kaggle CLI available")
    except Exception as e:
        log.error("Install Kaggle CLI: pip install kaggle"); raise

def kaggle_download(data_dir: Path, force=False) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "celeba-dataset.zip"
    if zip_path.exists() and not force:
        log.info(f"Found existing zip: {zip_path}"); return zip_path
    cmd = ["kaggle","datasets","download","-d",KAGGLE_DS,"-p",str(data_dir)]
    if force: cmd.append("--force")
    subprocess.check_call([c for c in cmd if c])
    zips = sorted(data_dir.glob("*.zip"), key=lambda p: p.stat().st_size, reverse=True)
    if not zips: raise FileNotFoundError("No zip found after Kaggle download.")
    if zips[0].name != "celeba-dataset.zip":
        shutil.move(str(zips[0]), str(zip_path))
    return zip_path

def extract_ann(zip_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path,"r") as z:
        members = z.namelist()
        def read(m): 
            with z.open(m) as f: return f.read()

        # identity
        id_txt = [m for m in members if m.endswith("identity_CelebA.txt")]
        id_csv = [m for m in members if m.endswith("identity_CelebA.csv")]
        if id_txt:
            (out_dir/"identity_CelebA.txt").write_text(read(id_txt[0]).decode("utf-8"))
        elif id_csv:
            import io
            df = pd.read_csv(io.BytesIO(read(id_csv[0])))
            cols = [c.lower() for c in df.columns]; df.columns = cols
            fname = "image_id" if "image_id" in cols else ("image" if "image" in cols else "filename")
            ident = "identity" if "identity" in cols else ("celebrity_id" if "celebrity_id" in cols else None)
            if not ident: raise ValueError(f"No identity column in {cols}")
            out = df[[fname, ident]].copy(); out.columns = ["filename","identity"]
            out.to_csv(out_dir/"identity_CelebA.txt", sep=" ", header=False, index=False)
        else:
            log.warning("identity_CelebA not found in zip; expecting you have it separately.")

        # partition
        part_txt = [m for m in members if m.endswith("list_eval_partition.txt")]
        part_csv = [m for m in members if m.endswith("list_eval_partition.csv")]
        if part_txt:
            (out_dir/"list_eval_partition.txt").write_text(read(part_txt[0]).decode("utf-8"))
        elif part_csv:
            import io
            dfp = pd.read_csv(io.BytesIO(read(part_csv[0])))
            cols = [c.lower() for c in dfp.columns]; dfp.columns = cols
            fname = "image_id" if "image_id" in cols else ("image" if "image" in cols else "filename")
            if "partition" not in cols: raise ValueError("partition column missing")
            outp = dfp[[fname, "partition"]].copy(); outp.columns = ["filename","partition"]
            outp.to_csv(out_dir/"list_eval_partition.txt", sep=" ", header=False, index=False)
        else:
            raise FileNotFoundError("list_eval_partition (csv/txt) not found in zip")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="./dataset")
    p.add_argument("--force-download", action="store_true")
    args = p.parse_args()
    ensure_kaggle()
    zip_path = kaggle_download(Path(args.data_dir), force=args.force_download)
    extract_ann(zip_path, Path(args.data_dir))
    print("\n✓ Annotations ready in ./dataset (TXT). Next: run download_data.py to materialize images.")
if __name__ == "__main__":
    main()