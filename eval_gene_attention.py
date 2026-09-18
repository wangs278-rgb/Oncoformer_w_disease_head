#!/usr/bin/env python3
"""Job C (run6): per-GENE and per-PATHOGENICITY attention allocation, done right.

The per-sample attention scalar (extract_gene_attn.py) is ~uniform/n -> dominated
by tumor size, so it can't answer "which genes get attention". Fix: for each real
gene token k, use RELATIVE attention

    rel[k] = cls_row[k] * n_keys      (n_keys = #valid keys incl CLS)

so rel>1 = above-uniform, rel<1 = below-uniform, independent of tumor size. Then
aggregate rel per gene id and per pathogenicity class (streaming). Also dumps a
per-sample gene-presence matrix for Jobs A/B (same val pass, aligned by sample_id).

No masking (attention on the real, unmasked tumor). run6/moco only.

    python eval_gene_attention.py --run run6

Writes <analysis>/plots/run6_gene_attention.npz  (per-gene / per-path rel sums+counts)
   and <analysis>/plots/run6_gene_sets.npz        (sample_id, presence[N,Ggenes], gene_names)
"""
import argparse, os, sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np

DATA_DIR = '/cv/home/wangs278/scratch/fmi'
BASE     = '/cv/scratch/u/wangs278/oncoformer_test'
RUNS = {'run6': dict(out='run6_moco_out', analysis='run6_moco_analysis',
                     cache=f'{DATA_DIR}/tokenized_dna_dx12_moco.pt')}

ap = argparse.ArgumentParser()
ap.add_argument('--run', default='run6', choices=list(RUNS))
args = ap.parse_args()
R = RUNS[args.run]

sys.path.insert(0, DATA_DIR)
import run6_moco_config
run6_moco_config.use_moco_oncoformer()
config = run6_moco_config.build_config(R['cache'])

import torch
import lightning as L
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')

import oncoformer
from oncoformer.dataset import OncoformerDataLoader
from oncoformer.models import OncoformerOmics
print(f'[{args.run}] oncoformer from {os.path.dirname(oncoformer.dataset.__file__)}')
assert 'Oncoformer_moco' in oncoformer.dataset.__file__

L.seed_everything(42, workers=True)
dl = OncoformerDataLoader(config)
tokenizers = dl.dataset.tokenizers
train_ids = set(map(str, getattr(dl.dataset, 'train_ids', []) or []))
val_ids   = set(map(str, getattr(dl.dataset, 'val_ids',   []) or []))
print(f'[{args.run}] val batches={len(dl.val_dataloader)} train={len(train_ids)} val={len(val_ids)}')

device = torch.device('cuda')
model = OncoformerOmics(config, tokenizers, checkpoint_dir=f'{DATA_DIR}/{R["out"]}/checkpoints')
ck = torch.load(f'{DATA_DIR}/{R["out"]}/checkpoints/last.ckpt', map_location='cpu', weights_only=False)
model.load_state_dict(ck.get('state_dict', ck), strict=False)
model = model.to(device).eval()
for m in model.modalities:
    st = model.streams[m]
    if hasattr(st, 'clear_attention_buffers'): st.clear_attention_buffers()
    if not getattr(st, 'attention_capture_enabled', False): st.enable_attention_capture()

# gene / pathogenicity vocab (ranked lists in vocab_dna.json metadata; offset by specials)
voc = json.load(open(f'{DATA_DIR}/vocab_dna.json'))
special = int(voc.get('special_token_len', 5))
gene_names = list(voc['metadata']['gene'])            # rank j -> id (special+j)
path_names = list(voc['metadata'].get('pathogenicity', []))
Gg = len(gene_names)
Gp = len(path_names) if path_names else int(voc['vocab_sizes'].get('pathogenicity', 9))

# streaming accumulators (flat, concatenated at end)
g_ids, g_rel, p_ids, p_rel = [], [], [], []
pres_rows = []      # per sample: array of gene columns present
ids_all, bait_all = [], []
n_captured_layers = None

for batch in dl.val_dataloader:
    batch = model.transfer_batch_to_device(batch, device, 0)
    unpacked = model._unpack_batch(batch)
    if len(unpacked) == 3:
        inputs, masks, graphs = unpacked
    else:
        inputs, masks = unpacked; graphs = None
    assert 'gene' in inputs['dna'], f"no 'gene' input; have {list(inputs['dna'])}"
    has_path = 'pathogenicity' in inputs['dna']

    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        H = model._encode_interleaved(inputs, masks, graphs) if graphs is not None \
            else model._encode_interleaved(inputs, masks)
    A = model._avg_self_attn('dna')
    if A is None:
        raise RuntimeError('attention buffers empty -> capture inactive')
    if n_captured_layers is None:
        n_captured_layers = sum(h.data is not None for h in model.streams['dna'].attn)
        print(f'[{args.run}] captured layers = {n_captured_layers} / {len(model.streams["dna"].attn)}')

    cls_row = A[:, 0, :].float()                    # [B, Lk]  CLS -> key
    mask = masks['dna'].float()                     # [B, Lk]  1 = valid key (incl CLS)
    nkeys = mask.sum(dim=1, keepdim=True).clamp_min(1)   # [B,1] valid keys incl CLS
    rel = cls_row * nkeys                            # [B, Lk]  >1 above uniform
    vg = masks['dna'].bool().clone(); vg[:, 0] = False   # valid non-CLS gene tokens
    genes = inputs['dna']['gene']                    # [B, Lk] long
    paths = inputs['dna']['pathogenicity'] if has_path else None

    vg_cpu = vg.cpu().numpy()
    genes_cpu = genes.cpu().numpy(); rel_cpu = rel.cpu().numpy()
    paths_cpu = paths.cpu().numpy() if has_path else None
    md = batch['sample_metadata']['sample_metadata']
    sids = list(md.index.astype(str)); baits = list(md['BaitSet'].astype(str))

    for b in range(vg_cpu.shape[0]):
        sel = vg_cpu[b]
        gid = genes_cpu[b][sel]
        rl  = rel_cpu[b][sel]
        g_ids.append(gid.astype(np.int64)); g_rel.append(rl.astype(np.float32))
        if paths_cpu is not None:
            p_ids.append(paths_cpu[b][sel].astype(np.int64)); p_rel.append(rl.astype(np.float32))
        # presence over gene rank columns [0,Gg). NOTE: gene token offset is 4
        # (tokenizer: id4=TP53=rank0), NOT the global special_token_len=5.
        cols = gid - 4
        cols = cols[(cols >= 0) & (cols < Gg)]
        pres_rows.append(np.unique(cols).astype(np.int32))
        ids_all.append(sids[b]); bait_all.append(baits[b])

# ---- val-only guard ----
ids_all = np.array(ids_all); bait_all = np.array(bait_all)
if val_ids:
    leaked = sum(1 for s in ids_all if s in train_ids)
    assert leaked == 0, f'{leaked} TRAIN samples leaked!'
    keep = np.array([s in val_ids for s in ids_all])
    print(f'[{args.run}] val-only: {int(keep.sum())}/{len(ids_all)} kept')
else:
    keep = np.ones(len(ids_all), bool)

# ---- per-gene / per-path aggregation (bincount) ----
g_ids = np.concatenate(g_ids); g_rel = np.concatenate(g_rel)
max_gid = int(g_ids.max())
gene_rel_sum = np.bincount(g_ids, weights=g_rel, minlength=max_gid + 1)
gene_rel_cnt = np.bincount(g_ids, minlength=max_gid + 1)
path_rel_sum = path_rel_cnt = None
if p_ids:
    p_ids = np.concatenate(p_ids); p_rel = np.concatenate(p_rel)
    mp = int(p_ids.max())
    path_rel_sum = np.bincount(p_ids, weights=p_rel, minlength=mp + 1)
    path_rel_cnt = np.bincount(p_ids, minlength=mp + 1)

plots = os.path.join(BASE, R['analysis'], 'plots'); os.makedirs(plots, exist_ok=True)
outa = os.path.join(plots, f'{args.run}_gene_attention.npz')
save = dict(gene_rel_sum=gene_rel_sum, gene_rel_cnt=gene_rel_cnt,
            gene_names=np.array(gene_names, dtype='U40'), special_token_len=np.int64(special),
            n_captured_layers=np.int64(n_captured_layers or 0))
if path_rel_sum is not None:
    save.update(path_rel_sum=path_rel_sum, path_rel_cnt=path_rel_cnt,
                path_names=np.array(path_names or [str(i) for i in range(Gp)], dtype='U40'))
np.savez_compressed(outa, **save)
print(f'[{args.run}] wrote {outa}  (genes with tokens: {(gene_rel_cnt>0).sum()})')

# ---- per-sample presence matrix (val-only) ----
idx_keep = np.where(keep)[0]
N = len(idx_keep)
presence = np.zeros((N, Gg), dtype=np.int8)
for r, i in enumerate(idx_keep):
    presence[r, pres_rows[i]] = 1
outs = os.path.join(plots, f'{args.run}_gene_sets.npz')
np.savez_compressed(outs, sample_id=ids_all[keep], baitset=bait_all[keep],
                    presence=presence, gene_names=np.array(gene_names, dtype='U40'))
print(f'[{args.run}] wrote {outs}  presence={presence.shape} mean_genes/sample={presence.sum(1).mean():.2f}')
