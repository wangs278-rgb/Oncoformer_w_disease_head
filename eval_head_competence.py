#!/usr/bin/env python3
"""Per-HEAD masked-prediction competence eval on the VAL split (run6, 7 heads).

run6's DNA MLM has 7 prediction heads (gene, alt_type, pathogenicity, zygosity,
aa_ref, aa_mut, aa_vaf_bin). This asks, per head: did it learn real signal, or is
it riding the class prior? For every masked position we record, per head:
    true id, pred id (argmax), p_true (softmax prob of the true class), p_max.
Downstream analyze_head_competence.py turns these into accuracy vs. majority-class
baseline (lift), macro-F1, calibration (ECE/NLL) and confusion matrices.

Faithfulness: reuses the model's OWN masking + forward path exactly as
OncoformerOmics._mlm_loss does (same mask mode, rate, tokenizer args) and iterates
apply_mlm_heads over ALL heads — so the gene head reproduces the logged
`val_mask_acc__dna__gene`. Only difference: we keep argmax + probs.

run6 ONLY: the existing eval uses the moco API (3-tuple _unpack_batch, extended
mask_tokens, apply_mlm_heads) which the base repo (run4/5) lacks. A base-API port
for the 2 shared heads is a separate follow-up.

    python eval_head_competence.py --run run6

Writes <analysis_dir>/plots/run6_head_eval.npz.
"""
import argparse, os, sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np

DATA_DIR = '/cv/home/wangs278/scratch/fmi'
BASE     = '/cv/scratch/u/wangs278/oncoformer_test'

RUNS = {
    'run6': dict(out='run6_moco_out',  analysis='run6_moco_analysis',
                 cache=f'{DATA_DIR}/tokenized_dna_dx12_moco.pt', moco=True,  dx12=True),
}

ap = argparse.ArgumentParser()
ap.add_argument('--run', default='run6', choices=list(RUNS))
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--passes', type=int, default=1,
                help='independent mask draws over the whole val set (aggregated)')
args = ap.parse_args()
R = RUNS[args.run]

# ---- config + correct oncoformer resolution (mirror eval_gene_stratified.py) ----
sys.path.insert(0, DATA_DIR)
assert R['moco'], 'this script is run6/moco only'
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
from oncoformer.training import mask_tokens
print(f'[{args.run}] oncoformer from {os.path.dirname(oncoformer.dataset.__file__)}')
assert 'Oncoformer_moco' in oncoformer.dataset.__file__

L.seed_everything(args.seed, workers=True)
dl = OncoformerDataLoader(config)
tokenizers = dl.dataset.tokenizers
print(f'[{args.run}] val batches = {len(dl.val_dataloader)}')

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

# ---- exact val masking config (mirror _mlm_loss / eval_gene_stratified) ----
tcfg = config['training']
t = 'dna'
modes_cfg = tcfg.get(f'{t}_val_mask_modes', tcfg.get(f'{t}_mask_modes', ['alterations']))
modes = list(modes_cfg if isinstance(modes_cfg, (list, tuple)) else [modes_cfg])
rate = float(tcfg.get(f'{t}_val_mask_rate', tcfg.get(f'{t}_masking_rate', 0.25)))
context_drop = float(tcfg.get('val_context_moddrop_p', tcfg.get('context_moddrop_p', 0.0)))
print(f'[{args.run}] mask modes={modes} rate={rate} context_drop={context_drop}')

t_tok    = model.tokenizers[t]
t_stream = model.streams[t]
enc_comps     = list(t_stream.modality_encoder.encoders.keys())
predict_heads = list(t_stream.mlm_heads.keys())
predict_alias = getattr(t_stream, 'predict_alias_map', {})
float_pads    = getattr(t_tok, 'component_pad_values', {})
head_vocab    = {c: int(t_stream.mlm_heads[c].out_features) for c in predict_heads}
assert 'gene' in predict_heads, f'no gene head; heads={predict_heads}'
print(f'[{args.run}] heads={predict_heads}  vocab={head_vocab}')


def to_dev(x):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: to_dev(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_dev(v) for v in x)
    return x


# per-head accumulators
acc = {c: dict(true=[], pred=[], ptrue=[], pmax=[]) for c in predict_heads}
n_batches = len(dl.val_dataloader)
for pass_i in range(args.passes):
    torch.manual_seed(args.seed + pass_i)
    for bi, batch in enumerate(dl.val_dataloader):
        batch = to_dev(batch)
        inputs_by_mod, masks_by_mod, graphs_by_mod = model._unpack_batch(batch)
        for mode in modes:
            mode = str(mode).lower()
            mwa = (mode == 'alterations')
            mwc = (mode == 'components')
            t_inputs = inputs_by_mod[t]
            t_masked, t_labels = mask_tokens(
                t_inputs, t_tok, device,
                mlm_probability=rate,
                mask_whole_alterations=mwa, mask_whole_components=mwc,
                encoded_components=enc_comps, predict_heads=predict_heads,
                predict_alias=predict_alias, float_pad_values=float_pads,
            )
            # every head must have a label tensor
            missing_lbl = [c for c in predict_heads if c not in t_labels]
            assert not missing_lbl, f'no labels for heads {missing_lbl}; got {list(t_labels)}'

            masked_inputs = {m: (t_masked if m == t else inputs_by_mod[m]) for m in model.modalities}
            ctx = {m: masks_by_mod[m] for m in model.modalities if m != t}
            from oncoformer.models import _maybe_moddrop
            ctx_dropped = _maybe_moddrop(ctx, p_drop=context_drop)
            fused_masks = {m: (masks_by_mod[m] if m == t else ctx_dropped[m]) for m in model.modalities}

            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                H = model._encode_interleaved(masked_inputs, fused_masks, graphs_by_mod)
                logits_dict = t_stream.apply_mlm_heads(H[t])       # {comp: [B, Lc, V]}

            for comp in predict_heads:
                V = head_vocab[comp]
                logits = logits_dict[comp].float().view(-1, logits_dict[comp].size(-1))  # [B*Lc, Vc]
                lab = t_labels[comp].view(-1)                                             # [B*Lc]
                valid = (lab != -100) & (lab >= 0) & (lab < V)                            # defensive
                if not valid.any():
                    continue
                lg = logits[valid]
                lab_v = lab[valid]
                probs = torch.softmax(lg, dim=-1)
                pred = probs.argmax(dim=-1)
                ptrue = probs.gather(1, lab_v.unsqueeze(1)).squeeze(1)
                pmax = probs.max(dim=-1).values
                acc[comp]['true'].append(lab_v.cpu().numpy().astype(np.int32))
                acc[comp]['pred'].append(pred.cpu().numpy().astype(np.int32))
                acc[comp]['ptrue'].append(ptrue.cpu().numpy().astype(np.float32))
                acc[comp]['pmax'].append(pmax.cpu().numpy().astype(np.float32))
        if (bi + 1) % 50 == 0:
            done = sum(len(a) for a in acc['gene']['true'])
            print(f'[{args.run}] pass {pass_i} batch {bi+1}/{n_batches} gene_masked={done}', flush=True)

# ---- gene id -> name map (same as eval_gene_stratified) ----
gene_names = None
special = int(json.load(open(f'{DATA_DIR}/vocab_dna.json')).get('special_token_len', 5))
try:
    voc = json.load(open(f'{DATA_DIR}/vocab_dna.json'))
    names = voc['metadata']['gene']
    gt = np.concatenate(acc['gene']['true']) if acc['gene']['true'] else np.array([0])
    gp = np.concatenate(acc['gene']['pred']) if acc['gene']['pred'] else np.array([0])
    max_id = int(max(gt.max(), gp.max()))
    arr = np.array(['<special_or_unk>'] * (max_id + 1), dtype=object)
    for i, nm in enumerate(names):
        tid = i + special
        if tid <= max_id:
            arr[tid] = nm
    gene_names = arr.astype('U40')
except Exception as e:
    print(f'[{args.run}] WARN gene name map: {e}')

save = dict(heads=np.array(predict_heads),
            epoch=np.int64(ck.get('epoch', -1) or -1),
            special_token_len=np.int64(special))
print(f'\n[{args.run}] === per-head masked counts + raw acc ===')
for comp in predict_heads:
    if not acc[comp]['true']:
        print(f'  {comp:16s} n=0 (no masked labels)'); continue
    tr = np.concatenate(acc[comp]['true']); pr = np.concatenate(acc[comp]['pred'])
    pt = np.concatenate(acc[comp]['ptrue']); pm = np.concatenate(acc[comp]['pmax'])
    save[f'{comp}__true']  = tr
    save[f'{comp}__pred']  = pr
    save[f'{comp}__ptrue'] = pt
    save[f'{comp}__pmax']  = pm
    save[f'{comp}__V']     = np.int64(head_vocab[comp])
    print(f'  {comp:16s} n={len(tr):>9d}  acc={float((tr==pr).mean()):.4f}  V={head_vocab[comp]}')
if gene_names is not None:
    save['gene_names'] = gene_names

plots = os.path.join(BASE, R['analysis'], 'plots')
os.makedirs(plots, exist_ok=True)
outp = os.path.join(plots, f'{args.run}_head_eval.npz')
np.savez_compressed(outp, **save)
print(f'[{args.run}] wrote {outp}')
