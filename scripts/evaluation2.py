import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
import librosa
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from pystoi import stoi  # New import for STOI

# Avoid symlink issues with SpeechBrain on Windows
os.environ.setdefault("SPEECHBRAIN_LOCAL_STRATEGY", "copy")

try:
    from speechbrain.pretrained import EncoderClassifier
except Exception as e:
    EncoderClassifier = None
    _speechbrain_import_error = e

# --- METRIC FUNCTIONS ---

def compute_mcd(ref_wav: np.ndarray, gen_wav: np.ndarray, sr: int) -> float:
    """Mel Cepstral Distortion (lower is better)."""
    ref_mfcc = librosa.feature.mfcc(y=ref_wav, sr=sr, n_mfcc=13)
    gen_mfcc = librosa.feature.mfcc(y=gen_wav, sr=sr, n_mfcc=13)
    
    _, path = fastdtw(ref_mfcc.T, gen_mfcc.T, dist=euclidean)
    
    dist = 0
    for i, j in path:
        diff = ref_mfcc[:, i] - gen_mfcc[:, j]
        dist += np.sqrt(np.sum(diff**2))
    
    mcd = (dist / len(path)) * (10 / np.log(10)) * np.sqrt(2)
    return float(mcd)

def compute_stoi(ref_wav: np.ndarray, gen_wav: np.ndarray, sr: int) -> float:
    """Short-Time Objective Intelligibility (higher is better, range 0-1)."""
    # STOI requires both signals to be the same length
    min_len = min(len(ref_wav), len(gen_wav))
    if min_len == 0:
        return float("nan")
    
    try:
        # extended=False is standard for most research papers
        score = stoi(ref_wav[:min_len], gen_wav[:min_len], sr, extended=False)
        return float(score)
    except Exception:
        return float("nan")

# --- CORE LOGIC ---

def list_audio_pairs(ref_dir: Path, gen_dir: Path, exts=(".wav", ".flac", ".mp3", ".m4a")) -> List[Tuple[Path, Path]]:
    ref_map: Dict[str, Path] = {}
    gen_map: Dict[str, Path] = {}
    for root, _, files in os.walk(ref_dir):
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in exts: ref_map[p.stem] = p
    for root, _, files in os.walk(gen_dir):
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in exts: gen_map[p.stem] = p
    common = sorted(set(ref_map.keys()) & set(gen_map.keys()))
    return [(ref_map[k], gen_map[k]) for k in common]

def load_audio(path: Path, target_sr: int, device: torch.device) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.dim() == 2 and wav.size(0) > 1: wav = wav.mean(0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)
    return wav.to(device)

def get_ecapa_encoder(device: torch.device):
    if EncoderClassifier is None:
        raise ImportError(f"speechbrain not available. Install via 'pip install speechbrain'")
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": str(device)},
        savedir=str(Path.home() / ".cache" / "speechbrain" / "spkrec-ecapa-voxceleb"),
    )

@torch.inference_mode()
def ecapa_embedding(encoder, wav: torch.Tensor) -> torch.Tensor:
    emb = encoder.encode_batch(wav)
    if isinstance(emb, (list, tuple)): emb = emb[0]
    while emb.dim() > 2: emb = emb.mean(dim=1)
    if emb.dim() == 1: emb = emb.unsqueeze(0)
    emb = emb.squeeze(0)
    return F.normalize(emb, dim=-1)

def cosine_similarity(e1: torch.Tensor, e2: torch.Tensor) -> float:
    return float(F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)).item())

def compute_eer(labels: List[int], scores: List[float]) -> Tuple[float, float]:
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2.0), float(thresholds[idx])

def compute_pesq_score(ref_wav: torch.Tensor, gen_wav: torch.Tensor, sr: int) -> float:
    from pesq import pesq as pesq_api
    ref = ref_wav.squeeze().cpu().numpy().astype("float32")
    deg = gen_wav.squeeze().cpu().numpy().astype("float32")
    L = min(len(ref), len(deg))
    if L <= 0: return float("nan")
    mode = "wb" if sr >= 16000 else "nb"
    try: return float(pesq_api(sr, ref[:L], deg[:L], mode))
    except: return float("nan")

def build_negative_pairs(pairs: List[Tuple[Path, Path]], negatives_per_sample: int = 1) -> List[Tuple[Path, Path]]:
    if len(pairs) < 2: return []
    refs = [p[0] for p in pairs]; gens = [p[1] for p in pairs]
    negs = []
    for i, g in enumerate(gens):
        choices = [r for j, r in enumerate(refs) if j != i]
        for r in random.sample(choices, k=min(negatives_per_sample, len(choices))): negs.append((r, g))
    return negs

def main():
    parser = argparse.ArgumentParser(description="Enhanced TTS Evaluation: MCD, STOI, ECAPA, PESQ")
    parser.add_argument("--ref_dir", type=str, required=True)
    parser.add_argument("--gen_dir", type=str, required=True)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compute_eer_random", action="store_true")
    parser.add_argument("--negatives_per_sample", type=int, default=1)
    parser.add_argument("--output", type=str, default="evaluation_results.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    sr = int(args.sr)
    pos_pairs = list_audio_pairs(Path(args.ref_dir), Path(args.gen_dir))
    neg_pairs = build_negative_pairs(pos_pairs, args.negatives_per_sample) if args.compute_eer_random else []

    encoder = get_ecapa_encoder(device)
    per_file, cos_pos, cos_all, labels_all, pesq_s, mcd_s, stoi_s = [], [], [], [], [], [], []

    print(f"Starting evaluation on {len(pos_pairs)} pairs...")

    for ref_p, gen_p in pos_pairs:
        ref_t = load_audio(ref_p, sr, device)
        gen_t = load_audio(gen_p, sr, device)
        
        ref_np = ref_t.squeeze().cpu().numpy()
        gen_np = gen_t.squeeze().cpu().numpy()

        cos = cosine_similarity(ecapa_embedding(encoder, ref_t), ecapa_embedding(encoder, gen_t))
        pesq_v = compute_pesq_score(ref_t, gen_t, sr)
        mcd_v = compute_mcd(ref_np, gen_np, sr)
        stoi_v = compute_stoi(ref_np, gen_np, sr)

        per_file.append({
            "ref": str(ref_p), "gen": str(gen_p),
            "cosine": cos, "pesq": pesq_v, "mcd": mcd_v, "stoi": stoi_v
        })
        cos_pos.append(cos); pesq_s.append(pesq_v); mcd_s.append(mcd_v); stoi_s.append(stoi_v)
        cos_all.append(cos); labels_all.append(1)

    for ref_p, gen_p in neg_pairs:
        cos = cosine_similarity(ecapa_embedding(encoder, load_audio(ref_p, sr, device)), 
                               ecapa_embedding(encoder, load_audio(gen_p, sr, device)))
        cos_all.append(cos); labels_all.append(0)

    results = {
        "speaker_similarity": {"mean_cosine": np.mean(cos_pos), "eer": compute_eer(labels_all, cos_all)[0] if neg_pairs else "nan"},
        "pesq": np.mean(pesq_s),
        "mcd": np.mean(mcd_s),
        "stoi": np.mean(stoi_s),
        "details": per_file,
        "meta": {"num_pos_pairs": len(pos_pairs), "sr": sr}
    }

    with open(args.output, "w") as f: json.dump(results, f, indent=2)

    print(f"\n--- Final Results ---")
    print(f"Mean Cosine: {results['speaker_similarity']['mean_cosine']:.4f}")
    print(f"PESQ: {results['pesq']:.4f}")
    print(f"MCD (Lower is Better): {results['mcd']:.4f}")
    print(f"STOI (Higher is Better): {results['stoi']:.4f}")

if __name__ == "__main__":
    main()