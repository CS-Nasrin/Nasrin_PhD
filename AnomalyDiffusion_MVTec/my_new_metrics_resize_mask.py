# eval_metrics_all.py
import argparse, os, shutil, tempfile, inspect, re
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
from PIL import Image
import torch

# ============================================================================
# Basic image I/O (kept from your version)
# ============================================================================
# --- Fix Windows double-counting in clean-fid: dedupe across glob calls ------
import os, glob, re

_glob_orig = glob.glob
_seen_by_dir = {}  # maps absolute dir -> set of normalized absolute file paths

def _glob_root(pattern: str) -> str:
    # directory portion before the first wildcard (* ? [)
    m = re.search(r'[\*\?\[]', pattern)
    base = pattern if m is None else pattern[:m.start()]
    d = os.path.dirname(base) or "."
    return os.path.normcase(os.path.abspath(d))

def _glob_dedupe_across_calls(pattern, recursive=False):
    res = _glob_orig(pattern, recursive=recursive)
    root = _glob_root(pattern)
    seen = _seen_by_dir.setdefault(root, set())
    out = []
    for p in res:
        key = os.path.normcase(os.path.abspath(p))  # case-insensitive on Windows
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out

glob.glob = _glob_dedupe_across_calls

def _glob_dedupe_reset():
    try:
        _seen_by_dir.clear()
    except NameError:
        pass
# -----------------------------------------------------------------------------



def _dedup_view(src_dir: Path) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="cf_dedup_"))
    # copy only files with standard image extensions, one copy each
    for p in src_dir.iterdir():
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
            (tmp / p.name).write_bytes(p.read_bytes())
    return tmp


def _debug_list_images_seen_by_cleanfid(dir_path: Path):
    # clean-fid accepts at least these (often includes .bmp too)
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    flat = [p for p in dir_path.glob("*") if p.suffix.lower() in exts]
    rec  = [p for p in dir_path.rglob("*") if p.suffix.lower() in exts]  # just in case there are nested subfolders
    print(f"[debug] {dir_path}  flat={len(flat)}  recursive={len(rec)}")
    if len(rec) != len(flat):
        print("        (There are images in subfolders; clean-fid may be counting them.)")
    # Show up to 40 filenames so you can spot the extras
    show = rec[:40]
    if show:
        print("        examples:", ", ".join(x.name for x in show))


def load_rgb(path):
    img = Image.open(path).convert('RGB')
    arr = np.array(img).astype(np.float32) / 255.0   # H,W,3 in [0,1]
    return torch.from_numpy(arr).permute(2,0,1)      # 3,H,W

def load_mask_resized(mask_path, target_wh):
    W, H = target_wh
    m = Image.open(mask_path).convert('L').resize((W, H), Image.NEAREST)
    m = (np.array(m).astype(np.float32) / 255.0 > 0.5).astype(np.float32)  # binarize 0/1
    return torch.from_numpy(m).unsqueeze(0)  # 1,H,W

def apply_mask(img_3chw, mask_1hw):
    # broadcasting applies the same 1-channel mask to all 3 channels
    return img_3chw * mask_1hw

def save_chw_png(img_3chw, out_path: Path):
    arr = (img_3chw.clamp(0,1).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(out_path)

# ============================================================================
# Utilities (listing, pairing, masking)
# ============================================================================

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

# --- Robust pairing helpers (numeric-aware, mask prefix/suffix tolerant) -----

def _gather_map(dir_path: Optional[Path], pattern_or: str) -> Dict[str, Path]:
    """{raw_stem: Path} for files matching '*.png|*.jpg|*.jpeg'."""
    m = {}
    if dir_path is None: return m
    patterns = [p.strip() for p in pattern_or.split("|") if p.strip()]
    for pat in patterns:
        for p in dir_path.glob(pat):
            m[p.stem] = p
    return m

def _norm_stem(s: str) -> str:
    """Strip common mask prefixes/suffixes so '000_mask', 'mask_000' -> '000'."""
    t = s.lower()
    for pre in ("mask_", "m_", "seg_", "bin_", "binary_", "ann_", "anno_"):
        if t.startswith(pre):
            t = t[len(pre):]
            break
    for suf in ("_mask", "-mask", ".mask", "_seg", "-seg", ".seg", "_m", "_bin", "_binary", "_anno", "_ann"):
        if t.endswith(suf):
            t = t[: -len(suf)]
            break
    return t

def _extract_num_key(s: str):
    """Return int from last digit run in normalized stem, or None."""
    t = _norm_stem(s)
    nums = re.findall(r'\d+', t)
    return int(nums[-1]) if nums else None

def _index_maps(name_to_path: Dict[str, Path]):
    """Build numeric-id map and normalized-stem map from {raw_stem: Path}."""
    id_map, stem_map = {}, {}
    for raw_stem, path in name_to_path.items():
        norm = _norm_stem(raw_stem)
        stem_map[norm] = path
        kid = _extract_num_key(raw_stem)
        if kid is not None and kid not in id_map:
            id_map[kid] = path
    return id_map, stem_map

def _match_pairs(img_map: Dict[str, Path], msk_map: Dict[str, Path]) -> List[Tuple[Path, Path]]:
    """Prefer numeric-ID overlap; fallback to normalized-stem overlap."""
    img_id, img_stem = _index_maps(img_map)
    msk_id, msk_stem = _index_maps(msk_map)
    inter_ids = sorted(set(img_id).intersection(msk_id))
    if inter_ids:
        return [(img_id[i], msk_id[i]) for i in inter_ids]
    inter_stems = sorted(set(img_stem).intersection(msk_stem))
    return [(img_stem[s], msk_stem[s]) for s in inter_stems]

# --- Simple mask utils (for older helpers) -----------------------------------

def _bin_mask(msk: Image.Image, thr: int = 128) -> np.ndarray:
    m = np.array(msk.convert("L"), dtype=np.uint8)
    m = (m >= thr).astype(np.uint8)
    return m

def _apply_mask_to_rgb(rgb: Image.Image, mask01: np.ndarray) -> Image.Image:
    arr = np.array(rgb.convert("RGB"), dtype=np.uint8)
    if mask01.ndim == 2:
        mask01 = mask01[..., None]  # HxWx1
    # Ensure size match; if mismatch, resize mask to image using NEAREST
    if (mask01.shape[0] != arr.shape[0]) or (mask01.shape[1] != arr.shape[1]):
        m_img = Image.fromarray((mask01.squeeze(-1)*255).astype(np.uint8))
        m_img = m_img.resize((arr.shape[1], arr.shape[0]), Image.NEAREST)
        mask01 = (np.array(m_img, dtype=np.uint8) >= 128)[..., None].astype(np.uint8)
    out = arr * mask01  # broadcast to 3 channels
    return Image.fromarray(out, mode="RGB")

# --- Higher-level masked set writers (still available) -----------------------

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
        out.save(out_dir / (img_p.stem + ".png"))
    return len(imgs), len(msks)

def _write_masked_set_and(
    real_img_dir: Path, real_msk_dir: Path,
    gen_img_dir: Path,  gen_msk_dir: Path,
    real_out: Path, gen_out: Path,
    img_glob: str, msk_glob: str
) -> int:
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

# ============================================================================
# Backends
# ============================================================================

def compute_cleanfid_unmasked(real_dir: Path, gen_dir: Path,
                              kid_subset_size: int, kid_subsets: int) -> Dict[str, float]:
    from cleanfid import fid
    _glob_dedupe_reset()
    # FID
    fid_val = fid.compute_fid(str(real_dir), str(gen_dir), mode="clean", num_workers=0, batch_size=32)
    _glob_dedupe_reset() 
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

# --- NEW: numeric-aware, mask-resized masked FID/KID -------------------------

def compute_cleanfid_masked(real_dir: Path, gen_dir: Path,
                            real_mask_dir: Optional[Path], gen_mask_dir: Optional[Path],
                            strategy: str, img_glob: str, msk_glob: str,
                            kid_subset_size: int, kid_subsets: int,
                            keep_masked: bool=False) -> Dict[str, float]:
    """
    Build masked image sets into temp folders and run clean-fid FID/KID on them.
    Masks are resized to each image size (NEAREST) and binarized to {0,1}.
    Matching handles 000.png ↔ 0.jpg and *_mask.png.
    """
    from cleanfid import fid

    def _mask_for_img(mask_path: Path, img_pil: Image.Image) -> np.ndarray:
        m = Image.open(mask_path).convert('L')
        if m.size != img_pil.size:
            m = m.resize(img_pil.size, Image.NEAREST)
        m = (np.array(m, dtype=np.float32) / 255.0 > 0.5).astype(np.float32)
        return m[..., None]  # (H,W,1)

    def _apply_and_save(img_path: Path, mask_path: Path, out_path: Path):
        img_pil = Image.open(img_path).convert('RGB')
        img = np.array(img_pil, dtype=np.float32) / 255.0      # (H,W,3)
        m   = _mask_for_img(mask_path, img_pil)                # (H,W,1)
        out = img * m
        out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(out).save(out_path)

    tmp_root = Path(tempfile.mkdtemp(prefix="masked_cleanfid_"))
    real_masked = tmp_root / "real_masked"
    gen_masked  = tmp_root / "gen_masked"
    real_masked.mkdir(parents=True, exist_ok=True)
    gen_masked.mkdir(parents=True, exist_ok=True)

    try:
        if strategy == "separate":
            if real_mask_dir is None or gen_mask_dir is None:
                raise ValueError("masked_strategy='separate' needs both real_mask_dir and gen_mask_dir.")

            # Real pairs
            r_imgs  = _gather_map(real_dir,      img_glob)
            r_masks = _gather_map(real_mask_dir, msk_glob)
            r_pairs = _match_pairs(r_imgs, r_masks)
            if not r_pairs:
                raise FileNotFoundError("[masked FID/KID] No REAL overlap. Check names & globs.")
            for ip, mp in r_pairs:
                _apply_and_save(ip, mp, real_masked / (ip.stem + ".png"))

            # Gen pairs
            g_imgs  = _gather_map(gen_dir,       img_glob)
            g_masks = _gather_map(gen_mask_dir,  msk_glob)
            g_pairs = _match_pairs(g_imgs, g_masks)
            if not g_pairs:
                raise FileNotFoundError("[masked FID/KID] No GEN overlap. Check names & globs.")
            for ip, mp in g_pairs:
                _apply_and_save(ip, mp, gen_masked / (ip.stem + ".png"))

        elif strategy == "and":
            if real_mask_dir is None or gen_mask_dir is None:
                raise ValueError("masked_strategy='and' needs both real_mask_dir and gen_mask_dir.")

            r_imgs  = _gather_map(real_dir,      img_glob)
            r_masks = _gather_map(real_mask_dir, msk_glob)
            g_imgs  = _gather_map(gen_dir,       img_glob)
            g_masks = _gather_map(gen_mask_dir,  msk_glob)

            r_iid, r_ist = _index_maps(r_imgs)
            r_mid, r_mst = _index_maps(r_masks)
            g_iid, g_ist = _index_maps(g_imgs)
            g_mid, g_mst = _index_maps(g_masks)

            ids_common = set(r_iid) & set(r_mid) & set(g_iid) & set(g_mid)
            if ids_common:
                for i in sorted(ids_common):
                    _apply_and_save(r_iid[i], r_mid[i], real_masked / (r_iid[i].stem + ".png"))
                    _apply_and_save(g_iid[i], g_mid[i], gen_masked  / (g_iid[i].stem + ".png"))
            else:
                stems_common = set(r_ist) & set(r_mst) & set(g_ist) & set(g_mst)
                if not stems_common:
                    raise FileNotFoundError("[masked FID/KID] No overlap across REAL/GEN imgs+masks (IDs or stems).")
                for s in sorted(stems_common):
                    _apply_and_save(r_ist[s], r_mst[s], real_masked / (r_ist[s].stem + ".png"))
                    _apply_and_save(g_ist[s], g_mst[s], gen_masked  / (g_ist[s].stem + ".png"))
        else:
            raise ValueError("strategy must be 'separate' or 'and'")

        if not any(real_masked.iterdir()) or not any(gen_masked.iterdir()):
            raise FileNotFoundError("Masked folders are empty. Check naming/globs for images vs masks.")

        _glob_dedupe_reset() 
        # Compute FID/KID on the masked folders
        fid_val = fid.compute_fid(str(real_masked), str(gen_masked),
                                  mode="clean", num_workers=0, batch_size=32)

        sig = inspect.signature(fid.compute_kid)
        kid_kwargs = dict(mode="clean", num_workers=0, batch_size=32)
        if "num_subsets" in sig.parameters:
            kid_kwargs["num_subsets"] = kid_subsets
        if "subset_size" in sig.parameters:
            kid_kwargs["subset_size"] = kid_subset_size

        _glob_dedupe_reset()
        kid_out = fid.compute_kid(str(real_masked), str(gen_masked), **kid_kwargs)
        if isinstance(kid_out, (tuple, list)) and len(kid_out) >= 1:
            kid_mean = float(kid_out[0])
            kid_std  = float(kid_out[1]) if len(kid_out) > 1 else None
        else:
            kid_mean, kid_std = float(kid_out), None

        return {
            "Masked_FID": float(fid_val),
            "Masked_KID_mean": kid_mean,
            "Masked_KID_std": kid_std,
            "kid_kwargs_used": kid_kwargs,
        }
    finally:
        if not keep_masked:
            shutil.rmtree(tmp_root, ignore_errors=True)

def compute_torchfidelity_is_unmasked(real_dir: Path, gen_dir: Path, splits: int) -> Dict[str, float]:
    from torch_fidelity import calculate_metrics
    m = calculate_metrics(
        #input1=str(real_dir), input2=str(gen_dir),
        input1=str(gen_dir),
        isc=True, fid=False, kid=False,
        isc_splits=splits,
        dataloader_num_workers=0, verbose=False
    )
    return {"IS_mean": float(m["inception_score_mean"]), "IS_std": float(m["inception_score_std"])}

# --- NEW: numeric-aware, mask-resized masked IS ------------------------------

def compute_torchfidelity_is_masked(gen_dir: Path, mask_dir: Optional[Path],
                                    strategy: str, real_dir: Optional[Path],
                                    real_mask_dir: Optional[Path],
                                    img_glob: str, msk_glob: str,
                                    splits: int, keep_masked: bool=False) -> Dict[str, float]:
    from torch_fidelity import calculate_metrics
    m = calculate_metrics(
    input1=str(gen_masked),        # <-- compute IS on masked GEN set
    isc=True, fid=False, kid=False,
    isc_splits=splits,
    dataloader_num_workers=0, verbose=False
)
    """
    Build a masked copy of the generated set, then run torch-fidelity IS on it.
    Masks are resized to each image size (NEAREST) and binarized to {0,1}.
    Matching handles 000.png ↔ 0.jpg and *_mask.png.
    """
    

    def _mask_for_img(mask_path: Path, img_pil: Image.Image) -> np.ndarray:
        m = Image.open(mask_path).convert('L')
        if m.size != img_pil.size:
            m = m.resize(img_pil.size, Image.NEAREST)
        m = (np.array(m, dtype=np.float32) / 255.0 > 0.5).astype(np.float32)
        return m[..., None]  # (H,W,1)

    def _apply_and_save(img_path: Path, mask_path: Path, out_path: Path):
        img_pil = Image.open(img_path).convert('RGB')
        img = np.array(img_pil, dtype=np.float32) / 255.0
        m   = _mask_for_img(mask_path, img_pil)
        #out = img * m
        #########################################grey 
        bg = np.array([0.5, 0.5, 0.5], dtype=np.float32)  # mid-gray in [0,1]
        out = img * m + bg * (1.0 - m) 
        #############################################
        out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(out).save(out_path)

    tmp_root = Path(tempfile.mkdtemp(prefix="masked_tfisc_"))
    gen_masked = tmp_root / "gen_masked"
    gen_masked.mkdir(parents=True, exist_ok=True)

    try:
        if strategy == "separate":
            if mask_dir is None:
                raise ValueError("Masked IS (separate) needs gen_mask_dir.")
            g_imgs  = _gather_map(gen_dir,  img_glob)
            g_masks = _gather_map(mask_dir, msk_glob)
            pairs = _match_pairs(g_imgs, g_masks)
            if not pairs:
                raise FileNotFoundError("[masked IS] No overlap gen images ↔ gen masks.")
            for ip, mp in pairs:
                _apply_and_save(ip, mp, gen_masked / (ip.stem + ".png"))

        elif strategy == "and":
            if real_dir is None or real_mask_dir is None or mask_dir is None:
                raise ValueError("'and' needs real_dir, real_mask_dir and gen_mask_dir.")
            r_imgs  = _gather_map(real_dir,      img_glob)
            r_masks = _gather_map(real_mask_dir, msk_glob)
            g_imgs  = _gather_map(gen_dir,       img_glob)
            g_masks = _gather_map(mask_dir,      msk_glob)

            r_iid, r_ist = _index_maps(r_imgs)
            r_mid, r_mst = _index_maps(r_masks)
            g_iid, g_ist = _index_maps(g_imgs)
            g_mid, g_mst = _index_maps(g_masks)

            ids_common = set(r_iid) & set(r_mid) & set(g_iid) & set(g_mid)
            if ids_common:
                for i in sorted(ids_common):
                    _apply_and_save(g_iid[i], g_mid[i], gen_masked / (g_iid[i].stem + ".png"))
            else:
                stems_common = set(r_ist) & set(r_mst) & set(g_ist) & set(g_mst)
                if not stems_common:
                    raise FileNotFoundError("[masked IS] No overlap across REAL/GEN imgs+masks (IDs or stems).")
                for s in sorted(stems_common):
                    _apply_and_save(g_ist[s], g_mst[s], gen_masked / (g_ist[s].stem + ".png"))
        else:
            raise ValueError("strategy must be 'separate' or 'and'")

        if not any(gen_masked.iterdir()):
            raise FileNotFoundError("Masked IS folder is empty. Check naming/globs for gen vs masks.")

        m = calculate_metrics(
            input1=str(real_dir or gen_dir),   # required by API; IS uses input2
            input2=str(gen_masked),
            isc=True, fid=False, kid=False,
            isc_splits=splits,
            dataloader_num_workers=0, verbose=False
        )
        return {
            "Masked_IS_mean": float(m["inception_score_mean"]),
            "Masked_IS_std":  float(m["inception_score_std"])
        }
    finally:
        if not keep_masked:
            shutil.rmtree(tmp_root, ignore_errors=True)

# --- LPIPS (your originals with a tiny robustness tweak in _apply_mask_to_rgb) --

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
    transforms_img = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t*2 - 1.0)
    ])
    transforms_msk = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),  # [1,H,W]
    ])

    def _m_for(k: str) -> Optional[torch.Tensor]:
        if strategy == "gen":
            if k not in mg_map: return None
            m = (transforms_msk(Image.open(mg_map[k]).convert("L")) >= 0.5).float()
            return m
        if strategy == "real":
            if k not in mr_map: return None
            m = (transforms_msk(Image.open(mr_map[k]).convert("L")) >= 0.5).float()
            return m
        # "and"
        if (k not in mg_map) or (k not in mr_map): return None
        mg = (transforms_msk(Image.open(mg_map[k]).convert("L")) >= 0.5).float()
        mr = (transforms_msk(Image.open(mr_map[k]).convert("L")) >= 0.5).float()
        return (mg * mr)

    loss = lpips.LPIPS(net=net).to(device).eval()
    import numpy as np
    scores = []
    with torch.no_grad():
        for k in keys:
            r = transforms_img(Image.open(rr[k]).convert("RGB")).unsqueeze(0).to(device)
            g = transforms_img(Image.open(rg[k]).convert("RGB")).unsqueeze(0).to(device)
            m = _m_for(k)
            if m is None:
                continue
            m = m.to(device)  # [1,H,W] in {0,1}
            m3 = m.repeat(1,3,1,1)
            scores.append(loss(r*m3, g*m3).item())
    return {"Masked_LPIPS_mean": float(np.mean(scores)) if scores else float("nan"),
            "pairs_used": len(scores), "strategy": strategy}

# ============================================================================
# CLI
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="Unified eval: clean-fid (FID/KID), torch-fidelity (IS), LPIPS (official) + masked variants.")

    # Your defaults (updated to include JPG/JPEG via glob args)
    ap.add_argument("--real_dir", default="C:/my_new_desktop/anomalydiffusion_original/data/wood/test/scratch")
    ap.add_argument("--gen_dir",  default="C:/my_new_desktop/anomalydiffusion_original/generated_dataset/wood/scratch/image")
    #ap.add_argument("--gen_dir", default="C:/my_new_desktop/anomalydiffusion_train/generated_dataset/broken_large/image")
    ap.add_argument("--real_mask_dir", default="C:/my_new_desktop/anomalydiffusion_original/data/wood/ground_truth/scratch")
    ap.add_argument("--gen_mask_dir",  default="C:/my_new_desktop/anomalydiffusion_original/generated_mask/wood/scratch")

    ap.add_argument("--img_glob", default="*.png|*.jpg|*.jpeg")
    ap.add_argument("--mask_glob", default="*.png|*.jpg|*.jpeg")

    ap.add_argument("--kid_subset_size", type=int, default=5)
    ap.add_argument("--kid_subsets", type=int, default=200)
    ap.add_argument("--isc_splits", type=int, default=1)
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

    real_dir_cf = _dedup_view(real_dir)
    gen_dir_cf  = _dedup_view(gen_dir)

    # --- Unmasked metrics ------------------------------------------------------
    
    print("\n== Unmasked metrics ==")
    _debug_list_images_seen_by_cleanfid(real_dir)
    _debug_list_images_seen_by_cleanfid(gen_dir)
    try:
        #cf = compute_cleanfid_unmasked(real_dir, gen_dir, args.kid_subset_size, args.kid_subsets)
        cf   = compute_cleanfid_unmasked(real_dir_cf, gen_dir_cf, args.kid_subset_size, args.kid_subsets)
        print(f"[clean-fid] FID: {cf['FID']:.6f} | KID_mean: {cf['KID_mean']:.6f} | KID_std: {cf['KID_std']}")
    except Exception as e:
        print(f"[clean-fid] Error: {e}")

    try:
        #tfis = compute_torchfidelity_is_unmasked(real_dir, gen_dir, args.isc_splits)
        tfis = compute_torchfidelity_is_unmasked(real_dir_cf, gen_dir_cf, args.isc_splits)
        print(f"[torch-fidelity] IS: {tfis['IS_mean']:.6f} ± {tfis['IS_std']:.6f} (splits={args.isc_splits})")
    except Exception as e:
        print(f"[torch-fidelity IS] Error: {e}")

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

        # Masked LPIPS (official)
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
