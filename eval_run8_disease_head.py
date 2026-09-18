#!/usr/bin/env python3
"""Supervised DiseaseTerm-head eval for run8_disease, on the VAL split, restricted to
the population the head was actually trained on:
  - BaitSet in {DX1, DX2}          (T7/D2 have no DNA -> degenerate embedding; excluded)
  - >0 real gene tokens            (guards DX1/DX2 samples with 0 mutations)
  - true DiseaseTerm is a SUPERVISED class (not matched by ignore_patterns=['other','nos'])

Why: the logged val_disease_term_loss is scored over ALL 4 baitsets (~1/3 T7/D2
degenerate) + class imbalance + label smoothing, so it is not interpretable. This
recomputes clean metrics on DX1/DX2 non-degenerate val samples.

Predictions/softmax are taken over the SUPERVISED class set only (ignored-class logits
masked to -inf), matching what the head was trained to output.

Metrics: acc@1, acc@5, macro/weighted F1, macro/micro/weighted OvR AUROC & AUPRC,
plus majority-class baseline. Runs for epoch=9 and epoch=19. Writes eval JSON.

Mirrors extract_disease_features.py for config/model/oncoformer resolution.
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
from collections import Counter

DATA_DIR = '/cv/home/wangs278/scratch/fmi'
OUT_DIR  = f'{DATA_DIR}/run8_disease_out'
CACHE    = f'{DATA_DIR}/tokenized_dna_dx12_moco.pt'
EVAL_DIR = f'{OUT_DIR}/eval'

sys.path.insert(0, DATA_DIR)
import run8_disease_config
run8_disease_config.use_moco_oncoformer()
config = run8_disease_config.build_config(CACHE)

import torch
import lightning as L
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')

import oncoformer
from oncoformer.dataset import OncoformerDataLoader
from oncoformer.models import OncoformerOmics, OncoformerPost
from torch.utils.data import DataLoader
assert 'Oncoformer_moco' in oncoformer.dataset.__file__, oncoformer.dataset.__file__

from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

os.makedirs(EVAL_DIR, exist_ok=True)

L.seed_everything(42, workers=True)
dl = OncoformerDataLoader(config, load_metadata=True)
tokenizers = dl.dataset.tokenizers
train_ids = set(map(str, getattr(dl.dataset, 'train_ids', []) or []))
val_ids   = set(map(str, getattr(dl.dataset, 'val_ids',   []) or []))
print(f'split: total={len(dl.dataset)} train={len(train_ids)} val={len(val_ids)} '
      f'disjoint={train_ids.isdisjoint(val_ids)}', flush=True)

# ---- vocab + EXACT training ignore rule (substring, lowercase; models.py:2761) ----
levels = json.load(open(f'{DATA_DIR}/vocab_metadata_disease_run8.json'))['disease_term']['levels']
term2id = {t: i for i, t in enumerate(levels)}
IGNORE = ['other', 'nos']
ignored_ids = set(i for i, l in enumerate(levels) if any(p in l.lower() for p in IGNORE))
sup_classes = np.array([i for i in range(len(levels)) if i not in ignored_ids])
n_cls = len(levels)
print(f'classes: total={n_cls} ignored={len(ignored_ids)} supervised={len(sup_classes)}', flush=True)

device = torch.device('cuda')
backbone = OncoformerOmics(config, tokenizers, checkpoint_dir=f'{OUT_DIR}/checkpoints')
model = OncoformerPost(backbone, config, checkpoint_dir=f'{OUT_DIR}/checkpoints')
assert 'disease_term' in model.prediction_heads

bs = int(config['training'].get('batch_size', 512))
val_loader = DataLoader(dl.val_dataset, batch_size=bs, shuffle=False, collate_fn=dl.collate_fn)


def to_dev(x):
    if torch.is_tensor(x): return x.to(device)
    if isinstance(x, dict): return {k: to_dev(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return type(x)(to_dev(v) for v in x)
    return x


@torch.no_grad()
def forward_val(ckpt_path):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ck.get('state_dict', ck)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f'loaded {os.path.basename(ckpt_path)} epoch={ck.get("epoch")} '
          f'missing={len(missing)} unexpected={len(unexpected)}', flush=True)
    model.to(device).eval()
    logits_all, baits, terms, ids, ngene = [], [], [], [], []
    for bi, batch in enumerate(val_loader):
        try:
            gtok = batch['omics_inputs']['dna']['gene'].long()
            ng = (gtok > 3).sum(dim=1).cpu().numpy()          # specials are ids 0..3
        except Exception:
            ng = None
        bd = to_dev(batch)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            lg = model(bd)['disease_term'].float().cpu().numpy()
        md = batch['sample_metadata']['sample_metadata']
        logits_all.append(lg)
        baits += list(md['BaitSet'].astype(str))
        terms += list(md['DiseaseTerm'].astype(str))
        ids   += list(md.index.astype(str))
        ngene.append(ng if ng is not None else np.full(lg.shape[0], -1))
        if (bi + 1) % 50 == 0:
            print(f'  {os.path.basename(ckpt_path)} batch {bi+1}/{len(val_loader)}', flush=True)
    logits = np.concatenate(logits_all, 0)
    baits = np.array(baits); terms = np.array(terms); ids = np.array(ids)
    ngene = np.concatenate(ngene, 0)
    n = min(len(logits), len(baits), len(terms), len(ids), len(ngene))
    return logits[:n], baits[:n], terms[:n], ids[:n], ngene[:n], int(ck.get('epoch', -1) or -1)


def softmax_supervised(logits):
    mask = np.full(logits.shape[1], -np.inf, dtype=np.float64)
    mask[sup_classes] = 0.0
    z = logits.astype(np.float64) + mask
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def compute_metrics(logits, y):
    probs = softmax_supervised(logits)
    order = np.argsort(-probs, axis=1)
    pred = order[:, 0]
    top5 = order[:, :5]
    acc1 = float((pred == y).mean())
    acc5 = float(np.any(top5 == y[:, None], axis=1).mean())
    maj = Counter(y.tolist()).most_common(1)[0][1] / len(y)

    present = np.unique(y)
    macro_f1 = float(f1_score(y, pred, labels=present, average='macro', zero_division=0))
    weighted_f1 = float(f1_score(y, pred, labels=present, average='weighted', zero_division=0))

    P = probs[:, present]
    Yoh = (y[:, None] == present[None, :]).astype(np.int8)
    aurocs, aps, supports = [], [], []
    for j in range(len(present)):
        yj = Yoh[:, j]; pj = P[:, j]; s = int(yj.sum())
        if s == 0 or s == len(yj):
            continue
        aurocs.append(roc_auc_score(yj, pj))
        aps.append(average_precision_score(yj, pj))
        supports.append(s)
    aurocs = np.array(aurocs); aps = np.array(aps); supports = np.array(supports)
    macro_auroc = float(aurocs.mean()); macro_auprc = float(aps.mean())
    w = supports / supports.sum()
    weighted_auroc = float((aurocs * w).sum()); weighted_auprc = float((aps * w).sum())
    micro_auroc = float(roc_auc_score(Yoh.ravel(), P.ravel()))
    micro_auprc = float(average_precision_score(Yoh.ravel(), P.ravel()))
    return dict(
        n=int(len(y)), n_classes_present=int(len(present)), n_classes_scored=int(len(aurocs)),
        majority_baseline_acc=float(maj),
        acc1=acc1, acc5=acc5, macro_f1=macro_f1, weighted_f1=weighted_f1,
        macro_auroc=macro_auroc, weighted_auroc=weighted_auroc, micro_auroc=micro_auroc,
        macro_auprc=macro_auprc, weighted_auprc=weighted_auprc, micro_auprc=micro_auprc,
    )


CKPTS = {9: f'{OUT_DIR}/checkpoints/epoch=epoch=9.ckpt',
         19: f'{OUT_DIR}/checkpoints/epoch=epoch=19.ckpt'}

results = {}
for tag, path in CKPTS.items():
    if not os.path.exists(path):
        print(f'SKIP epoch {tag}: {path} missing', flush=True); continue
    logits, baits, terms, ids, ngene, ep = forward_val(path)

    in_val = np.array([s in val_ids for s in ids])
    leaked = int(sum(1 for s in ids if s in train_ids))
    assert leaked == 0, f'{leaked} train ids leaked into val loader!'
    dx = np.isin(baits, ['DX1', 'DX2'])
    nz = (ngene > 0) if (ngene >= 0).any() else np.ones(len(ids), bool)
    true_id = np.array([term2id.get(t, -1) for t in terms])
    known = true_id >= 0
    supervised = ~np.isin(true_id, list(ignored_ids))
    keep = in_val & dx & nz & known & supervised

    # dedup by sample_id (keep first)
    kept_ids = ids[keep]
    _, uniq = np.unique(kept_ids, return_index=True)
    idx_keep = np.where(keep)[0][np.sort(uniq)]

    L_eval = logits[idx_keep]; y_eval = true_id[idx_keep]
    print(f'\nepoch {ep}: val={in_val.sum()} -> DX1/DX2={int((in_val&dx).sum())} '
          f'-> nondegenerate={int((in_val&dx&nz).sum())} -> supervised-label={len(y_eval)} '
          f'(deduped)', flush=True)
    m = compute_metrics(L_eval, y_eval)
    results[f'epoch{ep}'] = m
    print(f'epoch {ep} metrics: ' + '  '.join(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}'
                                              for k, v in m.items()), flush=True)

outp = f'{EVAL_DIR}/run8_disease_head_eval.json'
with open(outp, 'w') as f:
    json.dump({'population': 'val, BaitSet in DX1/DX2, >0 gene tokens, supervised label; '
                             'preds over supervised classes only',
               'n_supervised_classes': int(len(sup_classes)),
               'results': results}, f, indent=2)
print(f'\nWrote {outp}', flush=True)

# pretty table
print('\n=== SUMMARY (DX1/DX2 non-degenerate val, supervised classes) ===', flush=True)
hdr = ['epoch', 'n', 'maj_acc', 'acc@1', 'acc@5', 'macroF1', 'wF1',
       'macroAUROC', 'wAUROC', 'macroAUPRC', 'wAUPRC']
print('  '.join(f'{h:>10}' for h in hdr), flush=True)
for k, m in results.items():
    row = [k, m['n'], m['majority_baseline_acc'], m['acc1'], m['acc5'], m['macro_f1'],
           m['weighted_f1'], m['macro_auroc'], m['weighted_auroc'], m['macro_auprc'], m['weighted_auprc']]
    print('  '.join(f'{v:>10}' if isinstance(v, (int, str)) else f'{v:>10.4f}' for v in row), flush=True)
