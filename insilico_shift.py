#!/usr/bin/env python3
"""In-silico LINEAGE SHIFT: add one gene's modal mutation to patients of a SOURCE lineage
and measure how the disease head (and the embedding) moves toward a DEST lineage.

Recipients = DX1/DX2 patients whose TRUE DiseaseTerm == --source-term, that do NOT already
carry --gene and have a free slot. For each we record baseline vs after-addition:
  - pooled CLS embedding [512]  (before & after)   -> UMAP trajectory
  - P(source) and P(dest)       (before & after)   -> probability shift
  - disease-head argmax lineage (before & after)   -> flip source->dest

Reuses the exact KI insertion machinery from insilico_ki_disease.py (modal-token census +
free-slot insertion; positionless set-encoder makes this genuine presence).

Outputs run8_disease_out/shift/shift_<GENE>.npz (+ shift_<GENE>_meta.json).

Usage:
  python insilico_shift.py --gene CDH1 \
      --source-term "breast invasive ductal carcinoma (idc)" \
      --dest-term   "breast invasive lobular carcinoma (ilc)" [--max-batches N]
"""
import os, sys, json, argparse, warnings, hashlib
warnings.filterwarnings('ignore')
import numpy as np

DATA_DIR = '/cv/home/wangs278/scratch/fmi'
OUT_DIR  = f'{DATA_DIR}/run8_disease_out'
SHIFT_DIR = f'{OUT_DIR}/shift'
CACHE    = f'{DATA_DIR}/tokenized_dna_dx12_moco.pt'
CKPT     = f'{OUT_DIR}/checkpoints/epoch=epoch=19.ckpt'
CHUNK    = 512
SEED     = 42

ap = argparse.ArgumentParser()
ap.add_argument('--gene', required=True)
ap.add_argument('--source-term', required=True)
ap.add_argument('--dest-term', required=True)
ap.add_argument('--mode', choices=['ki', 'ko'], default='ki',
                help="ki: add gene to source patients -> shift to dest. "
                     "ko: remove gene from dest-lineage carriers -> shift back to source.")
ap.add_argument('--max-batches', type=int, default=0, help='0=all; >0 for a dry run')
args = ap.parse_args()

# output folder + shift direction (start -> target) depend on mode
SHIFT_DIR = f'{OUT_DIR}/shift' if args.mode == 'ki' else f'{OUT_DIR}/shift_ko'

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
assert 'Oncoformer_moco' in oncoformer.models.__file__, oncoformer.models.__file__

os.makedirs(SHIFT_DIR, exist_ok=True)
device = torch.device('cuda')

# --------------------------------------------------------------------------- setup
L.seed_everything(SEED, workers=True)
dl = OncoformerDataLoader(config, load_metadata=True)
tokenizers = dl.dataset.tokenizers

import pickle
tok = pickle.load(open(f'{DATA_DIR}/tokenizer_dna.pkl', 'rb'))
g2i = tok.component_token2idx['gene']
i2g = {v: k for k, v in g2i.items()}
REAL_MIN = 4
assert args.gene in g2i, f'{args.gene} not in gene vocab'
target_gid = int(g2i[args.gene])

levels = json.load(open(f'{DATA_DIR}/vocab_metadata_disease_run8.json'))['disease_term']['levels']
term2id = {t: i for i, t in enumerate(levels)}
IGNORE = ['other', 'nos']
ignored_ids = np.array([i for i, l in enumerate(levels) if any(p in l.lower() for p in IGNORE)])
sup_classes = np.array([i for i in range(len(levels)) if i not in set(ignored_ids.tolist())])
sup_names = np.array([levels[i] for i in sup_classes])
n_sup = len(sup_classes)
supid2col = {int(c): j for j, c in enumerate(sup_classes)}
for t in (args.source_term, args.dest_term):
    assert t in term2id and term2id[t] in supid2col, f'term not supervised: {t!r}'
# start_term = where recipients live; target_term = the lineage we test the shift toward.
# ki: source patients (add gene) -> dest.   ko: dest carriers (remove gene) -> source.
if args.mode == 'ki':
    start_term, target_term = args.source_term, args.dest_term
else:
    start_term, target_term = args.dest_term, args.source_term
start_col = supid2col[term2id[start_term]]
target_col = supid2col[term2id[target_term]]
verb = 'add to' if args.mode == 'ki' else 'remove from'
print(f'shift[{args.mode}]: {verb} {start_term!r} carriers of {args.gene}  ->  {target_term!r}',
      flush=True)

backbone = OncoformerOmics(config, tokenizers, checkpoint_dir=f'{OUT_DIR}/checkpoints')
model = OncoformerPost(backbone, config, checkpoint_dir=f'{OUT_DIR}/checkpoints')
assert 'disease_term' in model.prediction_heads
ck = torch.load(CKPT, map_location='cpu', weights_only=False)
missing, unexpected = model.load_state_dict(ck.get('state_dict', ck), strict=False)
assert len(missing) == 0 and len(unexpected) == 0, (missing[:5], unexpected[:5])
model.to(device).eval()
print(f'loaded {os.path.basename(CKPT)} epoch={ck.get("epoch")}', flush=True)

_negmask = torch.zeros(len(levels), device=device)
_negmask[torch.as_tensor(ignored_ids, device=device)] = float('-inf')
_sup_idx = torch.as_tensor(sup_classes, device=device)


def to_dev(x):
    if torch.is_tensor(x): return x.to(device, non_blocking=True)
    if isinstance(x, dict): return {k: to_dev(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return type(x)(to_dev(v) for v in x)
    return x


@torch.no_grad()
def probs_sup(batch):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits = model(batch)['disease_term'].float()
    z = logits + _negmask
    z = z - z.max(dim=1, keepdim=True).values
    e = torch.exp(z)
    return (e / e.sum(dim=1, keepdim=True)).index_select(1, _sup_idx)


@torch.no_grad()
def pooled_of(batch):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        return model.backbone.calculate_pooled_embedding(batch).float()


def sb(dna, m):
    return {'omics_inputs': {'dna': dna}, 'omics_masks': {'dna': m}, 'omics_graphs': {}}


# ---------- modal-token machinery (copied verbatim from insilico_ki_disease.py) ----------
COMP_KEYS = None
def set_comp_keys(dna):
    global COMP_KEYS
    if COMP_KEYS is None:
        COMP_KEYS = list(dna.keys()); assert 'gene' in COMP_KEYS

def slot_sig(dna_np, r, s):
    return hash(b'|'.join(dna_np[c][r, s].tobytes() for c in COMP_KEYS))

def census_target(loader, gid, max_batches):
    """Signature frequencies for ONE gene id (modal-token discovery)."""
    sig_count = {}
    for bi, batch in enumerate(loader):
        dna, mask = batch['omics_inputs']['dna'], batch['omics_masks']['dna']
        set_comp_keys(dna)
        dna_np = {c: dna[c].numpy() for c in COMP_KEYS}
        mk = mask.numpy(); gf = dna_np['gene']
        md = batch['sample_metadata']['sample_metadata']
        dx = md['BaitSet'].astype(str).isin(['DX1', 'DX2']).values
        for r in np.where(dx)[0]:
            for s in np.where((gf[r] == gid) & (mk[r] > 0))[0]:
                sig = slot_sig(dna_np, r, s)
                sig_count[sig] = sig_count.get(sig, 0) + 1
        if max_batches and bi + 1 >= max_batches: break
        if (bi + 1) % 200 == 0:
            print(f'  census batch {bi+1}/{len(loader)}  sigs={len(sig_count)}', flush=True)
    return sig_count

def capture_example(loader, want_sig, gid, max_batches):
    for bi, batch in enumerate(loader):
        dna, mask = batch['omics_inputs']['dna'], batch['omics_masks']['dna']
        dna_np = {c: dna[c].numpy() for c in COMP_KEYS}
        mk = mask.numpy(); gf = dna_np['gene']
        md = batch['sample_metadata']['sample_metadata']
        dx = md['BaitSet'].astype(str).isin(['DX1', 'DX2']).values
        for r in np.where(dx)[0]:
            for s in np.where((gf[r] == gid) & (mk[r] > 0))[0]:
                if slot_sig(dna_np, r, s) == want_sig:
                    return {c: dna[c][r, s].clone() for c in COMP_KEYS}
        if max_batches and bi + 1 >= max_batches: break
    return None

def insert_modal(dna, m_base, modal_dev):
    """Write the modal token into each row's first free slot, set mask=1. m_base not mutated."""
    K, Ln = m_base.shape
    f_idx = (m_base == 0).float().argmax(1)
    rows = torch.arange(K, device=device)
    dna_add = {c: dna[c].clone() for c in COMP_KEYS}
    m_add = m_base.clone()
    for c in COMP_KEYS:
        dna_add[c][rows, f_idx] = modal_dev[c]
    m_add[rows, f_idx] = 1
    return dna_add, m_add


# ------------------------------------------------- pass 1+C: modal token (KI mode only)
modal_dev = None
modal_frac = float('nan')
if args.mode == 'ki':
    print('\n===== pass 1: modal-token census for gene =====', flush=True)
    loader = DataLoader(dl.dataset, batch_size=CHUNK, shuffle=False, collate_fn=dl.collate_fn)
    sig_count = census_target(loader, target_gid, args.max_batches)
    assert sig_count, f'no real carriers of {args.gene} found'
    modal_sig, mcnt = max(sig_count.items(), key=lambda kv: kv[1])
    modal_frac = mcnt / sum(sig_count.values())
    print(f'{args.gene}: {len(sig_count)} distinct mutations, modal {mcnt}/{sum(sig_count.values())} '
          f'({100*modal_frac:.1f}%)', flush=True)
    print('===== pass C: capture modal token components =====', flush=True)
    loader = DataLoader(dl.dataset, batch_size=CHUNK, shuffle=False, collate_fn=dl.collate_fn)
    ex = capture_example(loader, modal_sig, target_gid, args.max_batches)
    assert ex is not None, 'failed to capture modal token'
    modal_dev = {c: ex[c].to(device) for c in COMP_KEYS}

# ------------------------------------------------- pass 2: baseline + perturbation
print(f'\n===== pass 2 [{args.mode}]: perturb {start_term!r} recipients =====', flush=True)
loader = DataLoader(dl.dataset, batch_size=CHUNK, shuffle=False, collate_fn=dl.collate_fn)
ids, emb_b, emb_p = [], [], []
pstart_b, pstart_p, ptgt_b, ptgt_p = [], [], [], []
arg_b, arg_p = [], []
n_recip = 0
for bi, batch in enumerate(loader):
    set_comp_keys(batch['omics_inputs']['dna'])          # ensure COMP_KEYS set (ko skips census)
    md = batch['sample_metadata']['sample_metadata']
    dx = md['BaitSet'].astype(str).isin(['DX1', 'DX2']).values
    terms = md['DiseaseTerm'].astype(str).values
    gf = batch['omics_inputs']['dna']['gene'].numpy()
    mk = batch['omics_masks']['dna'].numpy()
    sel = []
    for r in np.where(dx & (terms == start_term))[0]:
        present = set(np.unique(gf[r][(gf[r] >= REAL_MIN) & (mk[r] > 0)]).tolist())
        if args.mode == 'ki':
            if target_gid in present:              # must NOT already carry the gene
                continue
            if not (mk[r] == 0).any():             # needs a free slot to insert into
                continue
            if len(present) < 1:
                continue
        else:                                      # ko: must carry the gene, and have another
            if target_gid not in present:
                continue
            if len(present) < 2:                   # removal must not empty the sample
                continue
        sel.append(int(r))
    if sel:
        bd = to_dev(batch)
        inp, mask = bd['omics_inputs']['dna'], bd['omics_masks']['dna']
        ri = torch.as_tensor(sel, device=device)
        dna = {c: inp[c].index_select(0, ri) for c in COMP_KEYS}
        m_base = mask.index_select(0, ri)
        if args.mode == 'ki':
            md_dev = {c: modal_dev[c].unsqueeze(0).expand((len(sel),) + modal_dev[c].shape)
                      for c in COMP_KEYS}
            dna_p, m_p = insert_modal(dna, m_base, md_dev)
        else:
            dna_p = dna                            # same tokens; only the mask changes
            m_p = m_base.clone()
            m_p[dna['gene'] == target_gid] = 0     # remove the gene's slot(s)
        pb = probs_sup(sb(dna, m_base)); pa = probs_sup(sb(dna_p, m_p))
        eb = pooled_of(sb(dna, m_base)); ea = pooled_of(sb(dna_p, m_p))
        ids.extend(md.index.astype(str).values[sel].tolist())
        emb_b.append(eb.cpu().numpy().astype(np.float32)); emb_p.append(ea.cpu().numpy().astype(np.float32))
        pstart_b.extend(pb[:, start_col].cpu().numpy().tolist()); pstart_p.extend(pa[:, start_col].cpu().numpy().tolist())
        ptgt_b.extend(pb[:, target_col].cpu().numpy().tolist()); ptgt_p.extend(pa[:, target_col].cpu().numpy().tolist())
        arg_b.extend(sup_names[pb.argmax(1).cpu().numpy()].tolist())
        arg_p.extend(sup_names[pa.argmax(1).cpu().numpy()].tolist())
        n_recip += len(sel)
    if args.max_batches and bi + 1 >= args.max_batches: break
    if (bi + 1) % 100 == 0:
        print(f'  pass2 batch {bi+1}/{len(loader)}  recipients={n_recip}', flush=True)

emb_b = np.concatenate(emb_b, 0) if emb_b else np.zeros((0, 512), np.float32)
emb_p = np.concatenate(emb_p, 0) if emb_p else np.zeros((0, 512), np.float32)
arg_b = np.array(arg_b); arg_p = np.array(arg_p)
# generic schema: recipients START in source_term, shift toward dest_term (= target)
flip_to_tgt = float(np.mean((arg_p == target_term))) if n_recip else 0.0
was_tgt = float(np.mean((arg_b == target_term))) if n_recip else 0.0
np.savez_compressed(
    f'{SHIFT_DIR}/shift_{args.gene}.npz',
    sample_id=np.array(ids), emb_base=emb_b, emb_pert=emb_p,
    p_source_base=np.array(pstart_b, np.float32), p_source_pert=np.array(pstart_p, np.float32),
    p_dest_base=np.array(ptgt_b, np.float32), p_dest_pert=np.array(ptgt_p, np.float32),
    argmax_base=arg_b, argmax_pert=arg_p,
    gene=args.gene, source_term=start_term, dest_term=target_term,
)
meta = dict(gene=args.gene, mode=args.mode, source_term=start_term, dest_term=target_term,
            ki_source=args.source_term, ki_dest=args.dest_term,
            ckpt=os.path.basename(CKPT), n_recipients=int(n_recip), modal_frac=float(modal_frac),
            argmax_is_dest_before=was_tgt, argmax_is_dest_after=flip_to_tgt,
            mean_p_dest_before=float(np.mean(ptgt_b)) if n_recip else 0.0,
            mean_p_dest_after=float(np.mean(ptgt_p)) if n_recip else 0.0,
            mean_p_source_before=float(np.mean(pstart_b)) if n_recip else 0.0,
            mean_p_source_after=float(np.mean(pstart_p)) if n_recip else 0.0)
json.dump(meta, open(f'{SHIFT_DIR}/shift_{args.gene}_meta.json', 'w'), indent=2)
print(f'\nDONE [{args.mode}] {args.gene}  recipients={n_recip}', flush=True)
print(f'  {start_term!r} -> {target_term!r}', flush=True)
print(f'  argmax==target:  before={was_tgt:.3f}  after={flip_to_tgt:.3f}', flush=True)
print(f'  mean P(target): {meta["mean_p_dest_before"]:.3f} -> {meta["mean_p_dest_after"]:.3f}', flush=True)
print(f'  mean P(start) : {meta["mean_p_source_before"]:.3f} -> {meta["mean_p_source_after"]:.3f}', flush=True)
