#!/usr/bin/env python3
import argparse, os, io, tarfile, random, zlib
from pathlib import Path

def is_reg_file(ti: tarfile.TarInfo) -> bool:
    return ti.isreg()

def key_and_ext(name: str):
    p = Path(name)
    # ignore hidden tar members like PaxHeaders and directories
    if name.endswith("/"):
        return None, None
    stem = str(p.with_suffix(""))
    ext = p.suffix.lower()
    return stem, ext

def main():
    ap = argparse.ArgumentParser(description="Split a WebDataset-style tar of paired JSON+IMG into equal shards.")
    ap.add_argument("--input", required=True, help="Input .tar (uncompressed).")
    ap.add_argument("--out-dir", required=True, help="Output directory (will be created).")
    ap.add_argument("--num-shards", type=int, default=8, help="Number of shards to write (default: 8).")
    ap.add_argument("--samples-per-shard", type=int, default=None,
                    help="Samples per shard. If unset, computed as total//num_shards.")
    ap.add_argument("--img-exts", default=".png,.jpg,.jpeg", help="Comma-separated allowed image extensions.")
    ap.add_argument("--meta-ext", default=".json", help="Metadata extension (default: .json).")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle sample order before splitting (deterministic with --seed).")
    ap.add_argument("--seed", type=int, default=17, help="Shuffle seed (default: 17).")
    ap.add_argument("--prefix", default="eval", help="Output shard filename prefix (default: eval).")
    args = ap.parse_args()

    img_exts = {e.strip().lower() for e in args.img_exts.split(",") if e.strip()}
    meta_ext = args.meta_ext.lower()

    os.makedirs(args.out_dir, exist_ok=True)

    # Pass 1: index members by key (stem), preserving first-seen order
    items = {}   # key -> {"meta": TarInfo, "img": TarInfo}
    order = []   # keys in first-seen order
    with tarfile.open(args.input, "r") as tf:
        for ti in tf:
            if not is_reg_file(ti):
                continue
            key, ext = key_and_ext(ti.name)
            if key is None or ext is None:
                continue
            if ext not in img_exts and ext != meta_ext:
                continue
            slot = "meta" if ext == meta_ext else "img"
            d = items.get(key)
            if d is None:
                d = {}
                items[key] = d
                order.append(key)
            # prefer first occurrence if duplicates; warn by skipping subsequent
            if slot not in d:
                d[slot] = ti

    # Keep only complete pairs
    keys = [k for k in order if "meta" in items[k] and "img" in items[k]]
    missing = len(order) - len(keys)
    if missing:
        raise SystemExit(f"ERROR: {missing} samples are incomplete (missing meta or image). "
                         f"Only {len(keys)} complete pairs found.")

    total = len(keys)
    if args.shuffle:
        random.Random(args.seed).shuffle(keys)

    # Compute chunking
    num_shards = int(args.num_shards)
    if args.samples_per_shard is None:
        if total % num_shards != 0:
            raise SystemExit(f"ERROR: total samples {total} not divisible by num_shards={num_shards}. "
                             f"Pass --samples-per-shard to force a split.")
        sps = total // num_shards
    else:
        sps = int(args.samples_per_shard)
        if sps * num_shards > total:
            raise SystemExit(f"ERROR: requested {num_shards}×{sps}={num_shards*sps} > total {total}")
    if sps * num_shards != total:
        print(f"NOTE: {num_shards}×{sps}={num_shards*sps} < total {total}; extra samples will be ignored.")
        keys = keys[:num_shards*sps]

    # Pass 2: write shards, keeping JSON then IMG for each key
    pad = len(str(num_shards - 1))
    with tarfile.open(args.input, "r") as in_tf:
        for si in range(num_shards):
            start = si * sps
            end = start + sps
            shard_keys = keys[start:end]
            out_path = os.path.join(args.out_dir, f"{args.prefix}-{si:0{pad}d}.tar")
            count_pairs = 0
            with tarfile.open(out_path, "w") as out_tf:
                for k in shard_keys:
                    meta_ti = items[k]["meta"]
                    img_ti = items[k]["img"]

                    # write meta first
                    meta_f = in_tf.extractfile(meta_ti)
                    meta_bytes = meta_f.read() if meta_f is not None else b""
                    new_meta = tarfile.TarInfo(meta_ti.name)
                    new_meta.size = len(meta_bytes)
                    new_meta.mtime = meta_ti.mtime
                    out_tf.addfile(new_meta, io.BytesIO(meta_bytes))

                    # write image next
                    img_f = in_tf.extractfile(img_ti)
                    img_bytes = img_f.read() if img_f is not None else b""
                    new_img = tarfile.TarInfo(img_ti.name)
                    new_img.size = len(img_bytes)
                    new_img.mtime = img_ti.mtime
                    out_tf.addfile(new_img, io.BytesIO(img_bytes))

                    count_pairs += 1
            print(f"Wrote {out_path}  ({count_pairs} samples)")

    print(f"Done. Total written: {num_shards*sps} samples across {num_shards} shards "
          f"({sps} per shard).")

if __name__ == "__main__":
    main()
