# eval_metrics_all.py
import argparse, os, shutil, tempfile, inspect
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
from PIL import Image

# ---- Utilities ----------------------------------------------------------------

def _norm_key(path: Path, strip_mask_suffix: bool) -> str:
    """
    Normalize a filename for robust pairing:
    - optionally strip trailing '_mask'
    - drop leading zeros for purely numeric stems (e.g., '000' -> '0')
    """
    stem = path.stem
    if strip_mask_suffix and stem.endswith("_mask"):
        stem = stem[:-5]
    return str(int(stem)) if stem.isdigit() else stem

def _list_images(dir_path: Path, patterns: str = "*.png|*.jpg|*.jpeg") -> List[Path]:
    pats = [p.strip() for p in patterns.split("|")]
    out = []
    for p in pats:
        out.extend(sorted(dir_path.glob(p)))
        # also try upper-case extensions (Windows users sometimes mix cases)
        if p.lower().endswith(".jpg"): out.extend(sorted(dir_path.glob(p.upper())))
        if p.lower().endswith(".jpeg"): out.extend(sorted(dir_path.glob(p.upper())))
        if p.lower().endswith(".png"): out.extend(sorted(dir_path.glob(p.upper())))
    # unique while preserving order
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            uniq.append(x); seen.add(x)
    return uniq

def _align_images_and_masks(
    img_dir: Path, mask_dir: Path,
    img_glob="*.png|*.jpg|*.jpeg",
    mask_glob="*.png|*.jpg|*.jpeg",
) -> Tuple[List[Path], List[Path], List[str]]:
    imgs = _list_images(img_dir, img_glob)
    msks = _list_images(mask_dir, mask_glob)
    imap  = {_norm_key(p, False): p for p in imgs}
    mmap  = {_norm_key(p, True):  p for p in msks}
    keys  = sorted(set(imap) & set(mmap))
    return [imap[k] for k in keys], [mmap[k] for k in keys], keys

def _bin_mask(msk: Image.Image, thr: int = 128) -> np.ndarray:
    m = np.array(msk.convert("L"), dtype=np.uint8)
    m = (m >= thr).astype(np.uint8)
    return m

def _apply_mask_to_rgb(rgb: Image.Image, mask01: np.ndarray) -> Image.Image:
    arr = np.array(rgb.convert("RGB"), dtype=np.uint8)
    if mask01.ndim == 2:
        mask01 = mask01[..., None]  # HxWx1
    out = arr * mask01  # broadcast to 3 channels
    return Image.fromarray(out, mode="RGB")

def _write_masked_set_separate(
    img_dir: Path, msk_dir: Path, out_dir: Path,
    img_glob: str, msk_glob: str
) -> Tuple[int, int]:
    imgs, msks, keys = _align_images_and_masks(img_dir, msk_dir, img_glob, msk_glob)
    out_dir.mkdir(parents=True, exist_ok=True)
    for img_p, msk_p in zip(imgs, msks):
        m = _bin_mask(Image.open(msk_p))
        im = Image.open(img_p)
        out = _apply_mask_to_rgb(im, m)
        # save as png to avoid fresh lossy artifacts
        out.save(out_dir / (img_p.stem + ".png"))
    return len(imgs), len(msks)

def _write_masked_set_and(
    real_img_dir: Path, real_msk_dir: Path,
    gen_img_dir: Path,  gen_msk_dir: Path,
    real_out: Path, gen_out: Path,
    img_glob: str, msk_glob: str
) -> int:
    """
    Build aligned sets using intersection of (real_mask AND gen_mask) per common key.
    Only keys that exist in all four dirs are kept.
    """
    r_imgs = _list_images(real_img_dir, img_glob)
    g_imgs = _list_images(gen_img_dir,  img_glob)
    r_msks = _list_images(real_msk_dir, msk_glob)
    g_msks = _list_images(gen_msk_dir,  msk_glob)

    r_map  = {_norm_key(p, False): p for p in r_imgs}
    g_map  = {_norm_key(p, False): p for p in g_imgs}
    rm_map = {_norm_key(p, True):  p for p in r_msks}
    gm_map = {_norm_key(p, True):  p for p in g_msks}
    keys   = sorted(set(r_map) & set(g_map) & set(rm_map) & set(gm_map))

    real_out.mkdir(parents=True, exist_ok=True)
    gen_out.mkdir(parents=True, exist_ok=True)

    for k in keys:
        ri = Image.open(r_map[k])
        gi = Image.open(g_map[k])
        mr = _bin_mask(Image.open(rm_map[k]))
        mg = _bin_mask(Image.open(gm_map[k]))
        m  = (mr * mg).astype(np.uint8)
        _apply_mask_to_rgb(ri, m).save(real_out / (r_map[k].stem + ".png"))
        _apply_mask_to_rgb(gi, m).save(gen_out  / (g_map[k].stem + ".png"))
    return len(keys)

# ---- Backends -----------------------------------------------------------------

def compute_cleanfid_unmasked(real_dir: Path, gen_dir: Path,
                              kid_subset_size: int, kid_subsets: int) -> Dict[str, float]:
    from cleanfid import fid
    # FID
    fid_val = fid.compute_fid(str(real_dir), str(gen_dir), mode="clean", num_workers=0, batch_size=32)
    # KID (API changed across versions; build kwargs safely)
    sig = inspect.signature(fid.compute_kid)
    kid_kwargs = dict(mode="clean", num_workers=0, batch_size=32)
    if "num_subsets" in sig.parameters:
        kid_kwargs["num_subsets"] = kid_subsets
    if "subset_size" in sig.parameters:
        kid_kwargs["subset_size"] = kid_subset_size
    kid_val = fid.compute_kid(str(real_dir), str(gen_dir), **kid_kwargs)
    return {
        "FID": float(fid_val),
        "KID_mean": float(kid_val),
        "KID_std": None,  # clean-fid returns mean only
        "kid_kwargs_used": kid_kwargs,
    }

def compute_cleanfid_masked(real_dir: Path, gen_dir: Path,
                            real_mask_dir: Optional[Path], gen_mask_dir: Optional[Path],
                            strategy: str, img_glob: str, msk_glob: str,
                            kid_subset_size: int, kid_subsets: int,
                            keep_masked: bool=False) -> Dict[str, float]:
    from cleanfid import fid
    tmp_root = Path(tempfile.mkdtemp(prefix="masked_cleanfid_"))
    real_masked = tmp_root / "real_masked"
    gen_masked  = tmp_root / "gen_masked"
    try:
        if strategy == "separate":
            if real_mask_dir is None or gen_mask_dir is None:
                raise ValueError("masked_strategy='separate' needs both real_mask_dir and gen_mask_dir.")
            _ = _write_masked_set_separate(real_dir, real_mask_dir, real_masked, img_glob, msk_glob)
            _ = _write_masked_set_separate(gen_dir,  gen_mask_dir,  gen_masked,  img_glob, msk_glob)
        elif strategy == "and":
            if real_mask_dir is None or gen_mask_dir is None:
                raise ValueError("masked_strategy='and' needs both real_mask_dir and gen_mask_dir.")
            n = _write_masked_set_and(real_dir, real_mask_dir, gen_dir, gen_mask_dir,
                                      real_masked, gen_masked, img_glob, msk_glob)
            if n == 0:
                raise FileNotFoundError("No overlap across real/gen images+masks for 'and'.")
        else:
            raise ValueError("strategy must be 'separate' or 'and'")

        # now compute FID/KID using clean-fid on masked folders
        fid_val = fid.compute_fid(str(real_masked), str(gen_masked), mode="clean", num_workers=0, batch_size=32)
        sig = inspect.signature(fid.compute_kid)
        kid_kwargs = dict(mode="clean", num_workers=0, batch_size=32)
        if "num_subsets" in sig.parameters:
            kid_kwargs["num_subsets"] = kid_subsets
        if "subset_size" in sig.parameters:
            kid_kwargs["subset_size"] = kid_subset_size
        kid_val = fid.compute_kid(str(real_masked), str(gen_masked), **kid_kwargs)

        return {
            "Masked_FID": float(fid_val),
            "Masked_KID_mean": float(kid_val),
            "Masked_KID_std": None,
            "kid_kwargs_used": kid_kwargs,
        }
    finally:
        if not keep_masked:
            shutil.rmtree(tmp_root, ignore_errors=True)

def compute_torchfidelity_is_unmasked(real_dir: Path, gen_dir: Path, splits: int) -> Dict[str, float]:
    from torch_fidelity import calculate_metrics
    m = calculate_metrics(
        input1=str(real_dir), input2=str(gen_dir),
        isc=True, fid=False, kid=False,
        isc_splits=splits,
        dataloader_num_workers=0, verbose=False
    )
    return {"IS_mean": float(m["inception_score_mean"]), "IS_std": float(m["inception_score_std"])}

def compute_torchfidelity_is_masked(gen_dir: Path, mask_dir: Optional[Path],
                                    strategy: str, real_dir: Optional[Path],
                                    real_mask_dir: Optional[Path],
                                    img_glob: str, msk_glob: str,
                                    splits: int, keep_masked: bool=False) -> Dict[str, float]:
    """
    Build a masked copy of the *generated* set (optionally using intersection with real masks for 'and'),
    then run torch-fidelity IS on that masked folder.
    """
    from torch_fidelity import calculate_metrics
    tmp_root = Path(tempfile.mkdtemp(prefix="masked_tfisc_"))
    gen_masked = tmp_root / "gen_masked"
    try:
        if strategy == "separate":
            if mask_dir is None:
                raise ValueError("Masked IS (separate) needs gen_mask_dir.")
            _ = _write_masked_set_separate(gen_dir, mask_dir, gen_masked, img_glob, msk_glob)
        elif strategy == "and":
            if real_dir is None or real_mask_dir is None or mask_dir is None:
                raise ValueError("'and' needs real_dir, real_mask_dir and gen_mask_dir.")
            # 'and' here means: intersect real/gen masks and apply to GEN images only for IS
            # Build temp real/gen masked to get keys; then keep gen side.
            _ = _write_masked_set_and(real_dir, real_mask_dir, gen_dir, mask_dir,
                                      tmp_root / "real_tmp", gen_masked, img_glob, msk_glob)
        else:
            raise ValueError("strategy must be 'separate' or 'and'")

        m = calculate_metrics(
            input1=str(real_dir or gen_dir),  # not used by IS, but API requires something
            input2=str(gen_masked),
            isc=True, fid=False, kid=False,
            isc_splits=splits,
            dataloader_num_workers=0, verbose=False
        )
        return {"Masked_IS_mean": float(m["inception_score_mean"]), "Masked_IS_std": float(m["inception_score_std"])}
    finally:
        if not keep_masked:
            shutil.rmtree(tmp_root, ignore_errors=True)

def compute_lpips_unmasked(real_dir: Path, gen_dir: Path, net: str = "alex",
                           pairing: str = "filename", device: str = "cuda") -> Dict[str, float]:
    import torch, lpips
    from torchvision import transforms
    from PIL import Image

    def list_imgs(d: Path):
        return _list_images(d, "*.png|*.jpg|*.jpeg")

    R = list_imgs(real_dir); G = list_imgs(gen_dir)
    rr = {_norm_key(p, False): p for p in R}
    rg = {_norm_key(p, False): p for p in G}
    keys = sorted(set(rr) & set(rg))
    if not keys and pairing != "nearest":
        raise RuntimeError("LPIPS filename pairing found no overlapping basenames.")

    tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t*2 - 1.0)
    ])
    loss = lpips.LPIPS(net=net).to(device).eval()

    import numpy as np
    if keys:
        scores = []
        with torch.no_grad():
            for k in keys:
                r = tf(Image.open(rr[k]).convert("RGB")).unsqueeze(0).to(device)
                g = tf(Image.open(rg[k]).convert("RGB")).unsqueeze(0).to(device)
                scores.append(loss(r, g).item())
        return {"LPIPS_mean": float(np.mean(scores)), "pairs": len(keys), "pairing": "filename"}

    # fallback: nearest (each real to best gen)
    if pairing == "nearest":
        with torch.no_grad():
            Gt = [tf(Image.open(p).convert("RGB")).unsqueeze(0) for p in G]
            Gt = torch.cat(Gt, 0).to(device)
            scores = []
            for rp in R:
                r = tf(Image.open(rp).convert("RGB")).unsqueeze(0).to(device)
                dmin = None
                for i in range(0, Gt.shape[0], 32):
                    gb = Gt[i:i+32]
                    rb = r.expand(gb.shape[0], -1, -1, -1)
                    d  = loss(rb, gb).view(-1).cpu().numpy()
                    dmin = np.min(d) if dmin is None else min(dmin, np.min(d))
                scores.append(float(dmin))
        return {"LPIPS_mean": float(np.mean(scores)), "pairs": len(R), "pairing": "nearest"}

    raise RuntimeError("No LPIPS pairs.")

def compute_lpips_masked(real_dir: Path, gen_dir: Path,
                         real_mask_dir: Optional[Path], gen_mask_dir: Optional[Path],
                         strategy: str = "and", net: str = "alex", device: str = "cuda") -> Dict[str, float]:
    """
    Masked LPIPS: apply the mask (gen/real/intersection) at 256x256 before LPIPS.
    strategy in {"gen","real","and"}.
    """
    import torch, lpips
    from torchvision import transforms
    from PIL import Image

    R = _list_images(real_dir); G = _list_images(gen_dir)
    rr = {_norm_key(p, False): p for p in R}
    rg = {_norm_key(p, False): p for p in G}

    mg_map = {_norm_key(p, True): p for p in _list_images(gen_mask_dir)} if gen_mask_dir else {}
    mr_map = {_norm_key(p, True): p for p in _list_images(real_mask_dir)} if real_mask_dir else {}

    keys = sorted(set(rr) & set(rg))
    if not keys:
        raise RuntimeError("Masked LPIPS needs overlapping basenames.")

    # LPIPS-size transforms
    tf_img = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t*2 - 1.0)
    ])
    tf_msk = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),  # [1,H,W] after convert('L')
    ])

    def _m_for(k: str) -> Optional[torch.Tensor]:
        if strategy == "gen":
            if k not in mg_map: return None
            m = (tf_msk(Image.open(mg_map[k]).convert("L")) >= 0.5).float()
            return m
        if strategy == "real":
            if k not in mr_map: return None
            m = (tf_msk(Image.open(mr_map[k]).convert("L")) >= 0.5).float()
            return m
        # "and"
        if (k not in mg_map) or (k not in mr_map): return None
        mg = (tf_msk(Image.open(mg_map[k]).convert("L")) >= 0.5).float()
        mr = (tf_msk(Image.open(mr_map[k]).convert("L")) >= 0.5).float()
        return (mg * mr)

    loss = lpips.LPIPS(net=net).to(device).eval()
    import numpy as np
    scores = []
    with torch.no_grad():
        for k in keys:
            r = tf_img(Image.open(rr[k]).convert("RGB")).unsqueeze(0).to(device)
            g = tf_img(Image.open(rg[k]).convert("RGB")).unsqueeze(0).to(device)
            m = _m_for(k)
            if m is None:
                # if mask missing under chosen strategy, skip this pair
                continue
            m = m.to(device)  # [1,H,W] in {0,1}
            m3 = m.repeat(1,3,1,1)
            scores.append(loss(r*m3, g*m3).item())
    return {"Masked_LPIPS_mean": float(np.mean(scores)) if scores else float("nan"),
            "pairs_used": len(scores), "strategy": strategy}

# ---- CLI ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Unified eval: clean-fid (FID/KID), torch-fidelity (IS), LPIPS (official) + masked variants.")
    # ap.add_argument("--real_dir", required=True)
    # ap.add_argument("--gen_dir",  required=True)
    # ap.add_argument("--real_mask_dir", default=None)
    # ap.add_argument("--gen_mask_dir",  default=None)

    ap.add_argument("--real_dir", default="C:/my_new_desktop/anomalydiffusion_original/data/tile/test/gray_stroke")
    ap.add_argument("--gen_dir",  default="C:/my_new_desktop/anomalydiffusion_original/generated_dataset/generated_dataset/tile/gray_stroke/image")
    ap.add_argument("--real_mask_dir", default="C:/my_new_desktop/anomalydiffusion_original/data/tile/ground_truth/gray_stroke")
    ap.add_argument("--gen_mask_dir", default="C:/my_new_desktop/anomalydiffusion_original/generated_dataset/generated_dataset/tile/gray_stroke/mask")



    ap.add_argument("--img_glob", default="*.png|*.jpg|*.jpeg")
    ap.add_argument("--mask_glob", default="*.png|*.jpg|*.jpeg")

    ap.add_argument("--kid_subset_size", type=int, default=16)
    ap.add_argument("--kid_subsets", type=int, default=200)
    ap.add_argument("--isc_splits", type=int, default=5)
    ap.add_argument("--lpips_net", choices=["alex","vgg","squeeze"], default="alex")
    ap.add_argument("--lpips_pairing", choices=["filename","nearest"], default="filename")

    ap.add_argument("--masked_strategy", choices=["separate","and"], default="separate",
                    help="Mask protocol for FID/KID/IS (LPIPS masked uses gen/real/and internally)")

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--keep_masked", action="store_true", help="Keep temporary masked folders for inspection")

    args = ap.parse_args()
    real_dir = Path(args.real_dir); gen_dir = Path(args.gen_dir)
    real_mask_dir = Path(args.real_mask_dir) if args.real_mask_dir else None
    gen_mask_dir  = Path(args.gen_mask_dir)  if args.gen_mask_dir  else None

    # --- Unmasked metrics ------------------------------------------------------
    print("\n== Unmasked metrics ==")
    # clean-fid FID/KID
    try:
        cf = compute_cleanfid_unmasked(real_dir, gen_dir, args.kid_subset_size, args.kid_subsets)
        print(f"[clean-fid] FID: {cf['FID']:.6f} | KID_mean: {cf['KID_mean']:.6f} | KID_std: {cf['KID_std']}")
    except Exception as e:
        print(f"[clean-fid] Error: {e}")

    # torch-fidelity IS
    try:
        tfis = compute_torchfidelity_is_unmasked(real_dir, gen_dir, args.isc_splits)
        print(f"[torch-fidelity] IS: {tfis['IS_mean']:.6f} ± {tfis['IS_std']:.6f} (splits={args.isc_splits})")
    except Exception as e:
        print(f"[torch-fidelity IS] Error: {e}")

    # LPIPS (official)
    try:
        lp = compute_lpips_unmasked(real_dir, gen_dir, net=args.lpips_net, pairing=args.lpips_pairing, device=args.device)
        if "pairs" in lp:
            print(f"[LPIPS] {lp['LPIPS_mean']:.6f}  (pairs={lp['pairs']}, pairing={lp['pairing']})")
        else:
            print(f"[LPIPS] {lp['LPIPS_mean']:.6f}")
    except Exception as e:
        print(f"[LPIPS] Error: {e}")

    # --- Masked metrics --------------------------------------------------------
    if real_mask_dir or gen_mask_dir:
        print("\n== Masked metrics ==")

        # Masked FID/KID via clean-fid
        try:
            mcf = compute_cleanfid_masked(
                real_dir, gen_dir, real_mask_dir, gen_mask_dir,
                args.masked_strategy, args.img_glob, args.mask_glob,
                args.kid_subset_size, args.kid_subsets, keep_masked=args.keep_masked
            )
            print(f"[clean-fid] Masked FID: {mcf['Masked_FID']:.6f} | Masked KID_mean: {mcf['Masked_KID_mean']:.6f} | KID_std: {mcf['Masked_KID_std']}")
        except Exception as e:
            print(f"[clean-fid masked] Error: {e}")

        # Masked IS via torch-fidelity
        try:
            mtfis = compute_torchfidelity_is_masked(
                gen_dir, gen_mask_dir,
                args.masked_strategy, real_dir, real_mask_dir,
                args.img_glob, args.mask_glob,
                args.isc_splits, keep_masked=args.keep_masked
            )
            print(f"[torch-fidelity] Masked IS: {mtfis['Masked_IS_mean']:.6f} ± {mtfis['Masked_IS_std']:.6f} (splits={args.isc_splits})")
        except Exception as e:
            print(f"[torch-fidelity masked IS] Error: {e}")

        # Masked LPIPS (official) — choose strategy automatically:
        # if only gen masks -> 'gen'; only real masks -> 'real'; both -> 'and'
        try:
            if gen_mask_dir and not real_mask_dir:
                lpms = compute_lpips_masked(real_dir, gen_dir, None, gen_mask_dir, strategy="gen",
                                            net=args.lpips_net, device=args.device)
            elif real_mask_dir and not gen_mask_dir:
                lpms = compute_lpips_masked(real_dir, gen_dir, real_mask_dir, None, strategy="real",
                                            net=args.lpips_net, device=args.device)
            else:
                lpms = compute_lpips_masked(real_dir, gen_dir, real_mask_dir, gen_mask_dir, strategy="and",
                                            net=args.lpips_net, device=args.device)
            print(f"[LPIPS] Masked: {lpms['Masked_LPIPS_mean']:.6f}  (pairs_used={lpms['pairs_used']}, strategy={lpms['strategy']})")
        except Exception as e:
            print(f"[LPIPS masked] Error: {e}")
    else:
        print("\n(No mask dirs provided; masked metrics skipped.)")

if __name__ == "__main__":
    # Windows-safe: no multiprocessing spawned here
    main()
