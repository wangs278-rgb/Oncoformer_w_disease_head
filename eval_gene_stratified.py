#!/usr/bin/env python3
"""Gene-masked-prediction eval on the VAL split, for one run per process.

For every masked 'gene' position in the held-out val set, record (true_gene_id,
pred_gene_id). Downstream analysis (analyze_gene_stratified.py, CPU) turns this
into per-gene frequency vs. per-gene accuracy, baselines, and stratified bins to
answer: "is the ~0.13-0.20 gene accuracy just driven by high-frequency genes?"

Faithfulness: we reuse the model's OWN masking + forward path exactly as
OncoformerOmics.validation_step / _mlm_loss_mm do (same mask mode, same rate,
same tokenizer args), so the aggregate accuracy here reproduces the logged
`val_mask_acc__dna__gene`. Only difference: we keep the argmax predictions.

Run ONE run per process (run6's MoCo module-swap must not mix with fmi's
editable install):
    python eval_gene_stratified.py --run run4
    python eval_gene_stratified.py --run run5
    python eval_gene_stratified.py --run run6

Writes <analysis_dir>/plots/<run>_gene_eval.npz  (true, pred, gene_names, meta).
"""
import argparse, os, sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np

DATA_DIR = '/cv/home/wangs278/scratch/fmi'
BASE     = '/cv/scratch/u/wangs278/oncoformer_test'

RUNS = {
    'run4': dict(out='run4_fixed_out', analysis='run4_converge_analysis',
                 cache=f'{DATA_DIR}/tokenized_dna.pt',           moco=False, dx12=False),
    'run5': dict(out='run5_fixed_out', analysis='run5_dx12_analysis',
                 cache=f'{DATA_DIR}/tokenized_dna_dx12.pt',      moco=False, dx12=True),
    'run6': dict(out='run6_moco_out',  analysis='run6_moco_analysis',
                 cache=f'{DATA_DIR}/tokenized_dna_dx12_moco.pt', moco=True,  dx12=True),
}

ap = argparse.ArgumentParser()
ap.add_argument('--run', required=True, choices=list(RUNS))
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--passes', type=int, default=1,
                help='number of independent mask draws over the whole val set (aggregated)')
args = ap.parse_args()
R = RUNS[args.run]

# ---- config + correct oncoformer resolution (mirror extract_embeddings.py) ----
sys.path.insert(0, DATA_DIR)
if R['moco']:
    import run6_moco_config
    run6_moco_config.use_moco_oncoformer()
    config = run6_moco_config.build_config(R['cache'])
else:
    sys.path.insert(0, f'{DATA_DIR}/Oncoformer')
    import dx12_config
    config = dx12_config.build_config(R['cache'])
    if not R['dx12']:                       # run4 = all baitsets
        config['omics']['modalities']['dna']['data'].pop('baitset_filter', None)

import torch
import lightning as L
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')

import inspect
import oncoformer
from oncoformer.dataset import OncoformerDataLoader
from oncoformer.models import OncoformerOmics
from oncoformer.training import mask_tokens
print(f'[{args.run}] oncoformer from {os.path.dirname(oncoformer.dataset.__file__)}')
if R['moco']:
    assert 'Oncoformer_moco' in oncoformer.dataset.__file__

L.seed_everything(args.seed, workers=True)
dl = OncoformerDataLoader(config)
tokenizers = dl.dataset.tokenizers
print(f'[{args.run}] val batches = {len(dl.val_dataloader)}')

# val-only sanity: the val_dataloader IS the held-out split; confirm disjoint.
train_ids = set(map(str, getattr(dl.dataset, 'train_ids', []) or []))
val_ids   = set(map(str, getattr(dl.dataset, 'val_ids',   []) or []))
print(f'[{args.run}] split: total={len(dl.dataset)} train={len(train_ids)} '
      f'val={len(val_ids)} disjoint={train_ids.isdisjoint(val_ids)}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = OncoformerOmics(config, tokenizers, checkpoint_dir=f'{DATA_DIR}/{R["out"]}/checkpoints')
ckpt_path = f'{DATA_DIR}/{R["out"]}/checkpoints/last.ckpt'
ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
sd = ck.get('state_dict', ck)
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f'[{args.run}] loaded {ckpt_path} (epoch={ck.get("epoch")} '
      f'missing={len(missing)} unexpected={len(unexpected)})')
model.to(device).eval()

# ---- exact val masking config (mirror validation_step) ----
tcfg = config['training']
t = 'dna'
modes_cfg = tcfg.get(f'{t}_val_mask_modes', tcfg.get(f'{t}_mask_modes', ['alterations']))
modes = list(modes_cfg if isinstance(modes_cfg, (list, tuple)) else [modes_cfg])
rate = float(tcfg.get(f'{t}_val_mask_rate', tcfg.get(f'{t}_masking_rate', 0.25)))
context_drop = float(tcfg.get('val_context_moddrop_p', tcfg.get('context_moddrop_p', 0.0)))
print(f'[{args.run}] mask modes={modes} rate={rate} context_drop={context_drop}')

t_tok    = model.tokenizers[t]
t_stream = model.streams[t]
enc_comps    = list(t_stream.modality_encoder.encoders.keys())
predict_heads = list(t_stream.mlm_heads.keys())
predict_alias = getattr(t_stream, 'predict_alias_map', {})
float_pads    = getattr(t_tok, 'component_pad_values', {})
assert 'gene' in predict_heads, f'no gene head; heads={predict_heads}'

# Codebase-agnostic call shims: fmi's Oncoformer (run4/run5) and Oncoformer_moco
# (run6) differ in _unpack_batch arity, mask_tokens kwargs, and _encode_interleaved
# graphs arg. Detect each at runtime so one script serves both. (All runs are
# DNA-only => context modality-drop is a no-op, so it is omitted.)
_MT_PARAMS  = set(inspect.signature(mask_tokens).parameters)
_ENC_TAKES_GRAPHS = 'graphs' in inspect.signature(model._encode_interleaved).parameters
print(f'[{args.run}] api: mask_tokens_kwargs={"encoded_components" in _MT_PARAMS} '
      f'encode_takes_graphs={_ENC_TAKES_GRAPHS}')


def do_mask(t_inputs, mwa, mwc):
    kw = dict(mlm_probability=rate, mask_whole_alterations=mwa, mask_whole_components=mwc)
    if 'encoded_components' in _MT_PARAMS:
        kw.update(encoded_components=enc_comps, predict_heads=predict_heads,
                  predict_alias=predict_alias, float_pad_values=float_pads)
    return mask_tokens(t_inputs, t_tok, device, **kw)


def do_encode(masked_inputs, fused_masks, graphs):
    if _ENC_TAKES_GRAPHS:
        return model._encode_interleaved(masked_inputs, fused_masks, graphs)
    return model._encode_interleaved(masked_inputs, fused_masks)


def to_dev(x):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: to_dev(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_dev(v) for v in x)
    return x


true_all, pred_all = [], []
n_batches = len(dl.val_dataloader)
for pass_i in range(args.passes):
    torch.manual_seed(args.seed + pass_i)   # reproducible mask draw
    for bi, batch in enumerate(dl.val_dataloader):
        batch = to_dev(batch)
        up = model._unpack_batch(batch)
        if len(up) == 3:
            inputs_by_mod, masks_by_mod, graphs_by_mod = up
        else:
            inputs_by_mod, masks_by_mod = up; graphs_by_mod = {}
        for mode in modes:
            mode = str(mode).lower()
            mwa = (mode == 'alterations')
            mwc = (mode == 'components')
            t_inputs = inputs_by_mod[t]
            t_masked, t_labels = do_mask(t_inputs, mwa, mwc)
            masked_inputs = {m: (t_masked if m == t else inputs_by_mod[m]) for m in model.modalities}
            # DNA-only => no context modalities => fused masks == batch masks
            fused_masks = {m: masks_by_mod[m] for m in model.modalities}

            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                H = do_encode(masked_inputs, fused_masks, graphs_by_mod)
                gene_logits = t_stream.apply_mlm_heads(H[t])['gene']   # [B, Lc, V]

            labels = t_labels['gene'].view(-1)                        # [B*Lc]
            preds  = gene_logits.float().argmax(dim=-1).view(-1)      # [B*Lc]
            valid  = labels != -100
            if valid.any():
                true_all.append(labels[valid].cpu().numpy().astype(np.int32))
                pred_all.append(preds[valid].cpu().numpy().astype(np.int32))
        if (bi + 1) % 50 == 0:
            print(f'[{args.run}] pass {pass_i} batch {bi+1}/{n_batches} '
                  f'masked_so_far={sum(len(a) for a in true_all)}', flush=True)

true = np.concatenate(true_all)
pred = np.concatenate(pred_all)
acc = float((true == pred).mean())
print(f'[{args.run}] TOTAL masked gene positions={len(true)}  overall_acc={acc:.4f}')

# ---- gene id -> name map (align token ids to vocab_dna.json gene list) ----
gene_names = None
try:
    voc = json.load(open(f'{DATA_DIR}/vocab_dna.json'))
    names = voc['metadata']['gene']            # frequency-ranked list, 592 genes
    special = int(voc.get('special_token_len', 5))
    max_id = int(max(true.max(), pred.max()))
    arr = np.array(['<special_or_unk>'] * (max_id + 1), dtype=object)
    for i, nm in enumerate(names):
        tid = i + special
        if tid <= max_id:
            arr[tid] = nm
    gene_names = arr.astype('U40')
    # sanity: how many true labels fall in the gene id range [special, special+len)
    in_range = ((true >= special) & (true < special + len(names))).mean()
    print(f'[{args.run}] gene-id map: special={special} n_genes={len(names)} '
          f'true-in-generange={in_range:.3f}')
except Exception as e:
    print(f'[{args.run}] WARN could not build gene name map: {e}')

plots = os.path.join(BASE, R['analysis'], 'plots')
os.makedirs(plots, exist_ok=True)
outp = os.path.join(plots, f'{args.run}_gene_eval.npz')
save = dict(true=true, pred=pred, overall_acc=np.float32(acc),
            n_masked=np.int64(len(true)), epoch=np.int64(ck.get('epoch', -1) or -1),
            special_token_len=np.int64(int(json.load(open(f'{DATA_DIR}/vocab_dna.json')).get('special_token_len', 5))))
if gene_names is not None:
    save['gene_names'] = gene_names
np.savez_compressed(outp, **save)
print(f'[{args.run}] wrote {outp}')
