#!/usr/bin/env python3
"""VUS analysis extraction (run6). For every annotated variant token, read the
pathogenicity head's driver-likelihood and the variant's contextual vs sequence
embedding, so downstream analysis can (1) validate on known-vs-likely, (2) rank
VUS (unknown) by driver-likelihood, (3) test whether CONTEXT adds over sequence,
and (4) later join to AlphaMissense/ClinVar by variant identity.

pathogenicity ids (tokenizer): 4=known, 5=likely, 6=unknown(VUS). No masking —
pathogenicity is encode:false, so reading its head on the real tumor has no leak.
run6/moco only. val-only guard.

    python eval_vus.py --run run6

Writes:
  <plots>/run6_vus_scores.npz  (per token: label, p_known/likely/unknown, gene,
                                aa_pos, aa_ref, aa_mut, sample_id, baitset, group)
  <plots>/run6_vus_embeddings.npz (capped stratified subsample: ctx_emb, seq_emb,
                                label, gene, aa_pos, aa_ref, aa_mut  — float16)
"""
import argparse, os, sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np

DATA_DIR = '/cv/home/wangs278/scratch/fmi'
BASE     = '/cv/scratch/u/wangs278/oncoformer_test'
RUNS = {'run6': dict(out='run6_moco_out', analysis='run6_moco_analysis',
                     cache=f'{DATA_DIR}/tokenized_dna_dx12_moco.pt')}
PATH_IDS = {4: 0, 5: 1, 6: 2}          # known->0, likely->1, unknown(VUS)->2
CAP = 80000                            # per-class cap for stored embeddings
KEEP_P = 0.25

ap = argparse.ArgumentParser()
ap.add_argument('--run', default='run6', choices=list(RUNS))
args = ap.parse_args(); R = RUNS[args.run]

sys.path.insert(0, DATA_DIR)
import run6_moco_config
run6_moco_config.use_moco_oncoformer()
config = run6_moco_config.build_config(R['cache'])

import torch, lightning as L
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')
import oncoformer
from oncoformer.dataset import OncoformerDataLoader
from oncoformer.models import OncoformerOmics
assert 'Oncoformer_moco' in oncoformer.dataset.__file__

L.seed_everything(42, workers=True)
dl = OncoformerDataLoader(config)
train_ids = set(map(str, getattr(dl.dataset, 'train_ids', []) or []))
val_ids   = set(map(str, getattr(dl.dataset, 'val_ids',   []) or []))
print(f'[{args.run}] val batches={len(dl.val_dataloader)} val={len(val_ids)}')

device = torch.device('cuda')
model = OncoformerOmics(config, dl.dataset.tokenizers, checkpoint_dir=f'{DATA_DIR}/{R["out"]}/checkpoints')
ck = torch.load(f'{DATA_DIR}/{R["out"]}/checkpoints/last.ckpt', map_location='cpu', weights_only=False)
model.load_state_dict(ck.get('state_dict', ck), strict=False)
model = model.to(device).eval()
t_stream = model.streams['dna']
assert 'pathogenicity' in t_stream.mlm_heads, 'no pathogenicity head'

# accumulators (all-token scores) and capped embedding subsample
S = {k: [] for k in ('label', 'pk', 'pl', 'pu', 'gene', 'aapos', 'aaref', 'aamut', 'sid', 'bait', 'grp')}
E = {k: [] for k in ('ctx', 'seq', 'label', 'gene', 'aapos', 'aaref', 'aamut')}
ecap = {0: 0, 1: 0, 2: 0}
rng = np.random.default_rng(42)

for batch in dl.val_dataloader:
    batch = model.transfer_batch_to_device(batch, device, 0)
    unpacked = model._unpack_batch(batch)
    inputs, masks, graphs = unpacked if len(unpacked) == 3 else (unpacked[0], unpacked[1], None)
    for comp in ('pathogenicity', 'gene', 'aa_pos', 'aa_ref', 'aa_mut'):
        assert comp in inputs['dna'], f"missing input component {comp}"
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        seq = t_stream.initial_embed(inputs['dna'])                       # [B,L,512] pre-transformer
        H = model._encode_interleaved(inputs, masks, graphs) if graphs is not None \
            else model._encode_interleaved(inputs, masks)                 # contextual
        logits = t_stream.apply_mlm_heads(H['dna'])['pathogenicity']      # [B,L,V]
    ctx = H['dna']
    probs = torch.softmax(logits.float(), dim=-1)                         # [B,L,V]
    p3 = probs[..., [4, 5, 6]]                                            # known,likely,unknown
    p3 = p3 / p3.sum(dim=-1, keepdim=True).clamp_min(1e-9)                # renorm over 3 classes

    path = inputs['dna']['pathogenicity']                                 # [B,L]
    gene = inputs['dna']['gene']; aapos = inputs['dna']['aa_pos']
    aaref = inputs['dna']['aa_ref']; aamut = inputs['dna']['aa_mut']
    mk = masks['dna'].bool().clone(); mk[:, 0] = False                    # valid non-CLS
    annot = (path == 4) | (path == 5) | (path == 6)                      # annotated pathogenicity only
    valid = mk & annot

    md = batch['sample_metadata']['sample_metadata']
    sids = np.array(md.index.astype(str)); baits = np.array(md['BaitSet'].astype(str))
    grps = np.array(md['DiseaseGroup'].astype(str))

    v = valid.cpu().numpy()
    path_c = path.cpu().numpy(); gene_c = gene.cpu().numpy(); aapos_c = aapos.cpu().numpy()
    aaref_c = aaref.cpu().numpy(); aamut_c = aamut.cpu().numpy()
    pk = p3[..., 0].cpu().numpy(); pl = p3[..., 1].cpu().numpy(); pu = p3[..., 2].cpu().numpy()
    ctx_c = ctx.float().cpu().numpy(); seq_c = seq.float().cpu().numpy()

    B, Lk = v.shape
    for b in range(B):
        idxs = np.where(v[b])[0]
        for k in idxs:
            lab = PATH_IDS[int(path_c[b, k])]
            S['label'].append(lab); S['pk'].append(pk[b, k]); S['pl'].append(pl[b, k]); S['pu'].append(pu[b, k])
            S['gene'].append(int(gene_c[b, k])); S['aapos'].append(int(aapos_c[b, k]))
            S['aaref'].append(int(aaref_c[b, k])); S['aamut'].append(int(aamut_c[b, k]))
            S['sid'].append(sids[b]); S['bait'].append(baits[b]); S['grp'].append(grps[b])
            if ecap[lab] < CAP and rng.random() < KEEP_P:
                ecap[lab] += 1
                E['ctx'].append(ctx_c[b, k].astype(np.float16)); E['seq'].append(seq_c[b, k].astype(np.float16))
                E['label'].append(lab); E['gene'].append(int(gene_c[b, k])); E['aapos'].append(int(aapos_c[b, k]))
                E['aaref'].append(int(aaref_c[b, k])); E['aamut'].append(int(aamut_c[b, k]))

# ---- val-only guard (drop tokens from non-val samples) ----
sid_arr = np.array(S['sid'])
if val_ids:
    leaked = sum(1 for s in sid_arr if s in train_ids)
    assert leaked == 0, f'{leaked} train tokens leaked'
    keep = np.array([s in val_ids for s in sid_arr])
    print(f'[{args.run}] val-only: {int(keep.sum())}/{len(sid_arr)} tokens kept')
else:
    keep = np.ones(len(sid_arr), bool)

def col(name, dt):
    a = np.array(S[name]); return a[keep].astype(dt) if a.dtype != object else a[keep]

plots = os.path.join(BASE, R['analysis'], 'plots'); os.makedirs(plots, exist_ok=True)
np.savez_compressed(os.path.join(plots, f'{args.run}_vus_scores.npz'),
    label=col('label', np.int8), p_known=col('pk', np.float32), p_likely=col('pl', np.float32),
    p_unknown=col('pu', np.float32), gene=col('gene', np.int32), aa_pos=col('aapos', np.int32),
    aa_ref=col('aaref', np.int16), aa_mut=col('aamut', np.int16),
    sample_id=np.array(S['sid'])[keep], baitset=np.array(S['bait'])[keep], group=np.array(S['grp'])[keep])
lab = np.array(S['label'])[keep]
print(f'[{args.run}] scores: n={len(lab)}  known={int((lab==0).sum())} likely={int((lab==1).sum())} '
      f'unknown={int((lab==2).sum())}')

np.savez_compressed(os.path.join(plots, f'{args.run}_vus_embeddings.npz'),
    ctx_emb=np.array(E['ctx'], np.float16), seq_emb=np.array(E['seq'], np.float16),
    label=np.array(E['label'], np.int8), gene=np.array(E['gene'], np.int32),
    aa_pos=np.array(E['aapos'], np.int32), aa_ref=np.array(E['aaref'], np.int16),
    aa_mut=np.array(E['aamut'], np.int16))
print(f'[{args.run}] embeddings subsample: {dict((k,int(v)) for k,v in ecap.items())}')
print(f'[{args.run}] wrote run6_vus_scores.npz + run6_vus_embeddings.npz')
