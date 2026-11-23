"""Run two specified, both-present audio stress tests on the original frozen AVE models."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

SEEDS = (42, 43, 44)
HEADS = ("audio", "visual", "multimodal")
RATE = 48000
MANIFEST = Path("data/metadata/ave_manifest_validated.csv")
THRESHOLD = Path("results/final/hybrid_reliability_threshold_locked.json")
MAPPING = Path("results/baseline_clean/class_mapping.json")
REFERENCE = Path("results/final/hybrid_reliability_test_predictions.csv")
VISUAL = Path("data/embeddings/visual")
MODEL = "laion/clap-htsat-unfused"


def checkpoint(head, seed):
    return Path("models/baselines") / f"{head}_seed{seed}.pt"


def inspect_files():
    missing = [str(p) for p in (MANIFEST, THRESHOLD, MAPPING, REFERENCE)
               if not p.is_file()]
    missing += [str(checkpoint(h, s)) for h in HEADS for s in SEEDS
                if not checkpoint(h, s).is_file()]
    if not shutil.which("ffmpeg"):
        missing.append("ffmpeg on PATH")
    if not VISUAL.is_dir():
        missing.append(str(VISUAL))
    rows = []
    if MANIFEST.is_file():
        with MANIFEST.open(encoding="utf-8-sig", newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
        if len(rows) != 402 or len({r["sample_id"] for r in rows}) != 402:
            raise SystemExit("The manifest must contain the original 402 distinct test segments.")
        for label, paths in [
            ("test video", {Path(r["video_path"]) for r in rows}),
            ("visual embedding", {VISUAL / (r["sample_id"]+".npy") for r in rows})
        ]:
            absent = sorted(p for p in paths if not p.is_file())
            if absent:
                missing.append(f"{len(absent)} missing {label} files, beginning with {absent[0]}")
    if THRESHOLD.is_file() and float(json.loads(THRESHOLD.read_text())["selected_threshold"]) != 0.55:
        raise SystemExit("The saved threshold differs from 0.55. Restore the original file.")
    if missing:
        raise SystemExit("Restore the following inputs in the original experiment folder:\n  " + "\n  ".join(missing))
    return rows


def audio_conditions(wave):
    import numpy as np
    from scipy.signal import butter, sosfilt
    # Direct sound plus one 120-ms reflection, clipped to original segment duration.
    delay = round(0.120 * RATE)
    echo = wave.copy()
    echo[delay:] += 0.60 * wave[:-delay]
    echo /= max(1., float(np.abs(echo).max()))
    sos = butter(4, 2000, btype="lowpass", fs=RATE, output="sos")
    filtered = sosfilt(sos, wave).astype(np.float32)
    filtered /= max(1., float(np.abs(filtered).max()))
    return {"E0":wave, "echo":echo, "lowpass":filtered}


def extract(rows, device):
    import numpy as np
    import torch
    from transformers import ClapModel, ClapProcessor
    processor = ClapProcessor.from_pretrained(MODEL)
    model = ClapModel.from_pretrained(MODEL).eval().to(device)
    embeddings = {k:[] for k in ("E0","echo","lowpass")}
    for i, r in enumerate(rows):
        duration = max(float(r["end_time"])-float(r["start_time"]), 0.1)
        pcm = subprocess.check_output(["ffmpeg","-v","error","-ss",r["start_time"],"-i",r["video_path"],
                                       "-t",str(duration),"-vn","-ac","1","-ar",str(RATE),
                                       "-f","s16le","pipe:1"])
        wave = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
        if not wave.size:
            raise ValueError("Empty audio for " + r["sample_id"])
        seed = int.from_bytes(hashlib.sha256(r["sample_id"].encode()).digest()[:4],"little")
        for condition, corrupted in audio_conditions(wave).items():
            np.random.seed(seed)
            inputs = processor(audios=corrupted,sampling_rate=RATE,return_tensors="pt")
            inputs = {k:v.to(device) for k,v in inputs.items() if torch.is_tensor(v)}
            with torch.inference_mode():
                features = model.get_audio_features(**inputs)
                features = torch.nn.functional.normalize(features,p=2,dim=1)
            embeddings[condition].append(features.cpu().numpy()[0])
        if (i+1) % 25 == 0:
            print(f"Encoded {i+1} of {len(rows)}",flush=True)
    return {k:np.stack(v) for k,v in embeddings.items()}


def predict(x, head, seed, device):
    import torch
    from torch import nn
    ckpt = torch.load(checkpoint(head, seed), map_location="cpu", weights_only=True)
    model = nn.Module()
    model.network = nn.Sequential(nn.Linear(ckpt["input_dim"],256), nn.GELU(), nn.Dropout(0.30),
                                  nn.Linear(256,ckpt["num_classes"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    with torch.inference_mode():
        probs = model.network(torch.as_tensor(x,dtype=torch.float32,device=device)).softmax(1)
        confidence, label = probs.max(1)
    return label.cpu().numpy(), confidence.cpu().numpy()


def evaluate(rows, audio, device):
    import numpy as np
    import pandas as pd
    visual = np.stack([np.load(VISUAL / (r["sample_id"]+".npy")) for r in rows])
    mapping = json.loads(MAPPING.read_text())
    names = np.array(sorted(mapping, key=mapping.get))
    if sorted(mapping.values()) != list(range(28)):
        raise ValueError("Unexpected class mapping")
    parts = []
    for seed in SEEDS:
        vp, vc = predict(visual,"visual",seed,device)
        for cond, x in audio.items():
            ap, ac = predict(x,"audio",seed,device)
            bp, bc = predict(np.concatenate([x,visual],axis=1),"multimodal",seed,device)
            switch = abs(ac.astype(float)-vc.astype(float)) >= 0.55
            route = np.where(switch,np.where(ac>=vc,"audio","visual"),"fusion")
            hp = np.where(switch,np.where(ac>=vc,ap,vp),bp)
            hc = np.where(switch,np.where(ac>=vc,ac,vc),bc)
            df = pd.DataFrame(rows)[["sample_id","video_id","class_name"]].rename(columns={"class_name":"true_class"})
            df = df.assign(condition=cond, seed=seed, route=route, audio_prediction=names[ap], audio_confidence=ac,
                           visual_prediction=names[vp],visual_confidence=vc,baseline_prediction=names[bp],
                           baseline_confidence=bc, hybrid_prediction=names[hp],hybrid_confidence=hc)
            parts.append(df)
    return pd.concat(parts,ignore_index=True)


def summarize(data):
    import numpy as np
    import pandas as pd
    by_seed, intervals = [], []
    for name, condition in data.groupby("condition"):
        for seed, g in condition.groupby("seed"):
            metrics = {"condition":name,"seed":seed}
            for head in ("audio","visual","baseline","hybrid"):
                ok = g[head+"_prediction"].eq(g.true_class).to_numpy()
                conf = g[head+"_confidence"].to_numpy(dtype=float)
                if not np.isfinite(conf).all():
                    raise ValueError("A confidence is missing or nonfinite")
                bins = np.minimum((15*conf).astype(int),14)
                metrics[head+"_accuracy_pct"] = 100*ok.mean()
                metrics[head+"_ece_pct"] = 100*abs(np.bincount(bins,weights=ok-conf,minlength=15)).sum()/len(g)
            for route in ("audio","visual","fusion"):
                metrics["route_"+route+"_pct"] = 100*g.route.eq(route).mean()
            by_seed.append(metrics)
        difference = (condition.hybrid_prediction.eq(condition.true_class).astype(int) -
                      condition.baseline_prediction.eq(condition.true_class).astype(int))
        videos = difference.groupby(condition.video_id).agg(["sum","size"]).to_numpy()
        draws = np.random.default_rng(42).integers(len(videos),size=(10000,len(videos)))
        totals = videos[draws].sum(axis=1)
        lo,hi = np.percentile(100*totals[:,0]/totals[:,1],[2.5,97.5])
        intervals.append([name,100*difference.mean(),lo,hi])
    seeds = pd.DataFrame(by_seed)
    means = seeds.drop(columns="seed").groupby("condition").mean()
    ci = pd.DataFrame(intervals,columns=["condition","gain_pp","ci_low_pp","ci_high_pp"])
    return seeds,means.join(ci.set_index("condition")).reset_index()


def verify_clean(data):
    import pandas as pd
    keys = ["seed","sample_id"]
    ref = pd.read_csv(REFERENCE,dtype={"video_id":str})
    old = ref.loc[ref.experiment=="E0"].set_index(keys).sort_index()
    new = data.loc[data.condition=="E0"].set_index(keys).sort_index()
    if not new.index.equals(old.index):
        raise ValueError("E0 sample/seed identifiers differ from the archived run.")
    for field in ["video_id","true_class","audio_prediction","visual_prediction","baseline_prediction","hybrid_prediction"]:
        n = int((old[field]!=new[field]).sum())
        if n:
            raise ValueError(f"E0 mismatch in {field} for {n} rows. Check media, environment, and checkpoints before interpreting the new conditions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["check","run"])
    parser.add_argument("--repo", required=True, type=Path)
    args = parser.parse_args()
    os.chdir(args.repo.expanduser())
    rows = inspect_files()
    print("Found 402 test segments, video files, clean visual embeddings, nine heads, original E0 predictions and threshold 0.55.")
    if args.action == "run":
        import numpy as np
        import torch
        out = Path("results/unseen_audio_revision")
        if out.exists():
            raise SystemExit(f"{out} already exists. Preserve the previous run before continuing.")
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        data = evaluate(rows,extract(rows,device),device)
        verify_clean(data)
        seeds, means = summarize(data)
        with tempfile.TemporaryDirectory(dir="results") as temp:
            staged = Path(temp) / "output"
            staged.mkdir()
            data.to_csv(staged/"predictions.csv",index=False)
            seeds.to_csv(staged/"by_seed.csv",index=False)
            means.to_csv(staged/"summary.csv",index=False)
            staged.rename(out)
        print(means.to_string(index=False))
        print(f"Saved predictions and statistics to {out}")
