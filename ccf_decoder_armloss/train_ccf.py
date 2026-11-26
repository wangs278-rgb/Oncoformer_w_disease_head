#!/usr/bin/env python3
"""Train the CCFDecoder (ccf_model.py).  Same recipe/logic as run9 code/train_decoder.py (v2 recipe: run8 init, AdamW + warmup/cosine,
8 epochs, weight decay 0.5 on the per-variant table, bf16 autocast on GPU), data = ccf_data.CCFData.
Preemption/resume-safe: last.pt every --save-every steps, at epoch ends and on SIGTERM; the tie order of every batch is seeded by
(seed, epoch, batch), so a resumed run sees exactly the batches an uninterrupted run would.  Validation uses a fixed tie order.
Checkpoints: last.pt (final), best.pt (lowest val loss), epoch_<k>.pt every --keep-every epochs, metrics.csv.
Usage: python train_ccf.py --data-dir <run9 data> --ckpt-dir checkpoints --keep-every 1      (the arm-loss run; data dir also via $RUN9_DATA)"""
import argparse
import csv
import json
import math
import os
import signal
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ccf_data import CCFData, CCF_DIR, env_dir   # noqa: E402
from ccf_model import build_ccf_decoder          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--dry', action='store_true', help='2000 train / 500 val patients, 2 epochs')
ap.add_argument('--epochs', type=int, default=8)
ap.add_argument('--bs', type=int, default=256)
ap.add_argument('--lr', type=float, default=3e-4)
ap.add_argument('--wd', type=float, default=0.01)
ap.add_argument('--var-res-wd', type=float, default=0.5, help='weight decay on the per-variant residual table')
ap.add_argument('--warmup', type=float, default=0.02)
ap.add_argument('--dropout', type=float, default=0.1)
ap.add_argument('--clip', type=float, default=1.0)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--save-every', type=int, default=500)
ap.add_argument('--keep-every', type=int, default=0)
ap.add_argument('--eval-bs', type=int, default=512)
ap.add_argument('--no-run8-init', action='store_true')
ap.add_argument('--ckpt-dir', required=True)
ap.add_argument('--data-dir', default=None, help='run9 data dir (sequences.npz, vocab.npz, embeddings.pt, run8_init.pt); default $RUN9_DATA')
ap.add_argument('--ccf-dir', default=CCF_DIR, help='ccf_targets.npz + prefix_labels.npz; default $CCF_DATA or ./data')
ap.add_argument('--stop-after-steps', type=int, default=0, help='TEST ONLY: behave as if SIGTERM arrived after this many steps')
args = ap.parse_args()
args.data_dir = env_dir('RUN9_DATA', args.data_dir)
if args.dry:
    args.epochs, args.save_every = 2, 20

CK = args.ckpt_dir
os.makedirs(CK, exist_ok=True)
T0 = time.time()
CUDA = torch.cuda.is_available()
dev = torch.device('cuda' if CUDA else 'cpu')


def log(*a):
    print(f'[{time.time() - T0:8.1f}s]', *a, flush=True)


STOP = {'flag': False}


def _on_term(signum, frame):
    STOP['flag'] = True
    print(f'!!! signal {signum}: will checkpoint and exit at the next safe point', flush=True)


signal.signal(signal.SIGTERM, _on_term)
signal.signal(signal.SIGUSR1, _on_term)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if CUDA:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ----------------------------------------------------------------------------- data
D = CCFData(args.data_dir, args.ccf_dir)
vocab = D.vocab
emb = torch.load(f'{args.data_dir}/embeddings.pt', map_location='cpu')
init = None if args.no_run8_init else torch.load(f'{args.data_dir}/run8_init.pt', map_location='cpu')
train_ids = np.nonzero(D.split == 0)[0]
val_ids = np.nonzero(D.split == 1)[0]
if args.dry:
    train_ids, val_ids = train_ids[:2000], val_ids[:500]
steps_per_epoch = math.ceil(len(train_ids) / args.bs)
total_steps = steps_per_epoch * args.epochs
warmup_steps = max(1, int(args.warmup * total_steps))
N_PUR, N_SITE = len(D.pur_labels), len(D.site_labels)
log(f'train patients={len(train_ids)} val patients={len(val_ids)} steps/epoch={steps_per_epoch} total={total_steps} warmup={warmup_steps}; '
    f'purity classes={N_PUR} site classes={N_SITE}; device={dev}')


def to_dev(b):
    return {k: (v.pin_memory() if CUDA else v).to(dev, non_blocking=True) for k, v in b.items()}


def epoch_batches(epoch):
    perm = np.random.default_rng(args.seed * 1000 + epoch).permutation(train_ids)
    return [perm[i:i + args.bs] for i in range(0, len(perm), args.bs)]


def batch_rng(epoch, bi):
    return np.random.default_rng([args.seed, epoch, bi])


# sanity on one batch: the order contract of ccf_model.py
_b = D.collate(epoch_batches(0)[0], batch_rng(0, 0))
_G = D.G
_T = _b['gene'].shape[1]
_valid = torch.arange(_T)[None] < _b['n'][:, None]
_sep = _valid & (_b['gene'] == _G + 1)
assert (_sep.sum(1) == 1).all(), 'exactly one |U per patient'
_seg = torch.cumsum(_sep.long(), 1) - _sep.long()                              # 0 before |U, 1 after
_alt = _valid & ~_sep
_tm = _alt & (_seg == 0)
assert ((_b['cbin'] >= 0) == _tm).all(), 'cbin >= 0 exactly in the timed block'
_dd = (_b['cbin'][:, 1:] - _b['cbin'][:, :-1])[_tm[:, 1:] & _tm[:, :-1]]
assert (_dd <= 0).all(), 'timed block must be non-increasing in CCF bin'
_un = _alt & (_seg == 1)
_gd = (_b['gene'][:, 1:] - _b['gene'][:, :-1])[_un[:, 1:] & _un[:, :-1]]
assert (_gd >= 0).all(), 'untimed block must keep the frequency-rank order'
assert (_b['tok'][_alt] >= 0).all() and (torch.as_tensor(vocab['tok_gene'])[_b['tok'][_alt]] == _b['gene'][_alt]).all(), 'token / gene consistency'
assert (_b['n'] - _b['n_alt'] == 1).all()

# ----------------------------------------------------------------------------- model / optim
model = build_ccf_decoder(vocab, emb, init, dropout=args.dropout, n_sig=N_PUR, n_site=N_SITE).to(dev)
n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
log(f'CCF decoder params (trainable) = {n_par / 1e6:.2f}M; run8 init = {init is not None}')
decay, no_decay, vres = [], [], []
for name, p in model.named_parameters():
    if name == 'var_res' and args.var_res_wd > 0:
        vres.append(p)
    elif p.ndim < 2 or name.startswith(('disease_emb', 'sig_emb', 'site_emb', 'pos_emb', 'gene_emb', 'var_res', 'sep_emb')):
        no_decay.append(p)
    else:
        decay.append(p)
groups = [{'params': decay, 'weight_decay': args.wd}, {'params': no_decay, 'weight_decay': 0.0}]
if vres:
    groups.append({'params': vres, 'weight_decay': args.var_res_wd})
opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.98), eps=1e-8)


def lr_at(step):
    if step < warmup_steps:
        return args.lr * (step + 1) / warmup_steps
    frac = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return args.lr * (0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, frac))))


def fwd(b):
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=CUDA):
        return model(b)


@torch.no_grad()
def evaluate(ids):
    model.eval()
    tot = {k: 0.0 for k in ('lg', 'lv', 'lc', 'ag', 'av', 'ac', 'ng', 'nv', 'nc')}
    for ci, i in enumerate(range(0, len(ids), args.eval_bs)):
        if STOP['flag']:
            model.train()
            return None
        b = to_dev(D.collate(ids[i:i + args.eval_bs], np.random.default_rng([args.seed, 10_000_019, ci])))
        o = fwd(b)
        ng, nv, nc = float(o['n_gene']), float(o['n_var']), float(o['n_ccf'])
        tot['lg'] += float(o['loss_gene']) * ng
        tot['lv'] += float(o['loss_var']) * nv
        tot['lc'] += float(o['loss_ccf']) * nc
        tot['ag'] += float(o['acc_gene']); tot['av'] += float(o['acc_var']); tot['ac'] += float(o['acc_ccf'])
        tot['ng'] += ng; tot['nv'] += nv; tot['nc'] += nc
    model.train()
    r = {'val_loss_gene': tot['lg'] / tot['ng'], 'val_loss_var': tot['lv'] / max(1, tot['nv']), 'val_loss_ccf': tot['lc'] / max(1, tot['nc']),
         'val_acc_gene': tot['ag'] / tot['ng'], 'val_acc_var': tot['av'] / max(1, tot['nv']), 'val_acc_ccf': tot['ac'] / max(1, tot['nc'])}
    r['val_loss'] = r['val_loss_gene'] + r['val_loss_var'] + r['val_loss_ccf']
    return r


def save_ckpt(path, epoch, batch_idx, global_step, best):
    state = {'model': model.state_dict(), 'opt': opt.state_dict(), 'epoch': epoch, 'batch_idx': batch_idx, 'global_step': global_step,
             'best_val': best, 'args': vars(args), 'torch_rng': torch.get_rng_state()}
    if CUDA:
        state['cuda_rng'] = torch.cuda.get_rng_state()
    tmp = path + '.tmp'
    torch.save(state, tmp)
    os.replace(tmp, path)


LAST, BEST, METRICS = f'{CK}/last.pt', f'{CK}/best.pt', f'{CK}/metrics.csv'
start_epoch, start_batch, global_step, best_val = 0, 0, 0, float('inf')
if os.path.exists(LAST):
    st = torch.load(LAST, map_location='cpu', weights_only=False)
    model.load_state_dict(st['model'])
    opt.load_state_dict(st['opt'])
    start_epoch, start_batch, global_step, best_val = st['epoch'], st['batch_idx'], st['global_step'], st['best_val']
    torch.set_rng_state(st['torch_rng'])
    if CUDA and 'cuda_rng' in st:
        torch.cuda.set_rng_state(st['cuda_rng'])
    for k in ('bs', 'epochs', 'lr', 'seed', 'dropout', 'no_run8_init', 'var_res_wd', 'wd', 'dry'):
        if st['args'].get(k) != vars(args).get(k):
            log(f'WARNING resume arg mismatch {k}: ckpt={st["args"].get(k)} now={vars(args).get(k)}')
    log(f'RESUMED from {LAST}: epoch={start_epoch} batch={start_batch} step={global_step} best={best_val:.4f}')
else:
    r0 = evaluate(val_ids)
    if r0 is None:
        sys.exit(0)
    log('initial val (before training):', json.dumps({k: round(v, 4) for k, v in r0.items()}))

fields = ['epoch', 'global_step', 'lr', 'train_loss', 'train_loss_gene', 'train_loss_var', 'train_loss_ccf', 'val_loss', 'val_loss_gene',
          'val_loss_var', 'val_loss_ccf', 'val_acc_gene', 'val_acc_var', 'val_acc_ccf', 'epoch_time_s']
if not os.path.exists(METRICS):
    with open(METRICS, 'w', newline='') as fh:
        csv.writer(fh).writerow(fields)

model.train()
for epoch in range(start_epoch, args.epochs):
    batches = epoch_batches(epoch)
    te = time.time()
    run = {'l': 0.0, 'lg': 0.0, 'lv': 0.0, 'lc': 0.0, 'k': 0}
    for bi in range(start_batch, len(batches)):
        b = to_dev(D.collate(batches[bi], batch_rng(epoch, bi)))
        lr = lr_at(global_step)
        for gr in opt.param_groups:
            gr['lr'] = lr
        o = fwd(b)
        loss = o['loss']
        if not torch.isfinite(loss):
            raise FloatingPointError(f'non-finite loss at step {global_step}: {o}')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        global_step += 1
        run['l'] += float(loss); run['lg'] += float(o['loss_gene']); run['lv'] += float(o['loss_var']); run['lc'] += float(o['loss_ccf']); run['k'] += 1
        if global_step % 100 == 0:
            k = run['k']
            log(f'ep{epoch} b{bi + 1}/{len(batches)} step{global_step} lr={lr:.2e} loss={run["l"] / k:.4f} '
                f'(gene {run["lg"] / k:.4f} var {run["lv"] / k:.4f} ccf {run["lc"] / k:.4f}) gnorm={float(gn):.2f} '
                f'{(bi + 1 - start_batch) / (time.time() - te):.1f} it/s')
        if args.stop_after_steps and global_step == args.stop_after_steps:
            STOP['flag'] = True
        stop_now = STOP['flag']
        if global_step % args.save_every == 0 or stop_now:
            save_ckpt(LAST, epoch, bi + 1, global_step, best_val)
        if stop_now:
            log(f'checkpointed at epoch={epoch} batch={bi + 1}; exiting for requeue')
            sys.exit(0)
    k = run['k'] if run['k'] > 0 else float('nan')
    save_ckpt(LAST, epoch, len(batches), global_step, best_val)
    r = evaluate(val_ids)
    if r is None:
        log('stopped during validation; exiting for requeue (epoch will re-validate on resume)')
        sys.exit(0)
    row = {'epoch': epoch, 'global_step': global_step, 'lr': lr_at(global_step), 'train_loss': run['l'] / k, 'train_loss_gene': run['lg'] / k,
           'train_loss_var': run['lv'] / k, 'train_loss_ccf': run['lc'] / k, 'epoch_time_s': time.time() - te, **r}
    with open(METRICS, 'a', newline='') as fh:
        csv.writer(fh).writerow([row.get(f, '') for f in fields])
    log(f'EPOCH {epoch} ' + json.dumps({k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in row.items()}))
    if r['val_loss'] < best_val:
        best_val = r['val_loss']
        torch.save({'model': model.state_dict(), 'epoch': epoch, 'global_step': global_step, 'val': r, 'args': vars(args), 'data_dir': args.data_dir}, BEST + '.tmp')
        os.replace(BEST + '.tmp', BEST)
        log(f'  new best val_loss={best_val:.4f} -> {BEST}')
    if args.keep_every and (epoch + 1) % args.keep_every == 0:
        torch.save({'model': model.state_dict(), 'epoch': epoch, 'global_step': global_step, 'val': r, 'args': vars(args), 'data_dir': args.data_dir},
                   f'{CK}/epoch_{epoch}.pt')
        log(f'  kept {CK}/epoch_{epoch}.pt')
    save_ckpt(LAST, epoch + 1, 0, global_step, best_val)
    start_batch = 0
log(f'TRAINING DONE best val_loss={best_val:.4f}')
