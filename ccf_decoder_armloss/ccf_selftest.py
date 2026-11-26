#!/usr/bin/env python3
"""CCFDecoder self-tests on toy data (CPU, fp32, dropout 0; toy from run9 code/selftest.py:make_toy).
 C1 collate == independent construction: timed block non-increasing in bin, |U after it, untimed in original order, multiset kept;
    same rng -> same batch, other rng -> other tie order, tie order uniform within a tie group
 C2 forward losses/accuracies == brute-force per-step loop with an independent grammar (rank order only in the untimed block,
    END only after |U, |U only before it, CCF bin <= previous timed bin)
 C3 causality of the hidden states
 C4 greedy generate == argmax of the teacher-forced logits on the generated sequence
 C5 sampled patients obey the grammar (descending bins, cbin >= 0 only in the timed block, rank order in the untimed block,
    no repeated specific variant, token/gene consistency, truncation at the cap)
 C6 capped patient (127 alterations): no END target; 129 positions fit
 C7 prefix: purity and site change the output, NO_SITE row is zero, site_emb receives gradient
 C8 train_ccf.py end-to-end on toy data; a run stopped mid-epoch and resumed ends with weights identical to an uninterrupted run
Usage: python3 ccf_selftest.py"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ccf_model import CCFDecoder, MAX_ALTS, N_BIN, ccf_bin_rep    # noqa: E402
from ccf_data import CCFData                                      # noqa: E402


def make_toy(rng, G=8, per_gene=6, n_freq=4, n_pat=3000, d=512, n_levels=5):
    """Toy vocab / sequences / embeddings; copied from run9 code/selftest.py:make_toy (rng passed in instead of a global)."""
    NC = G * per_gene
    conc_gene_idx = np.repeat(np.arange(G), per_gene)
    # tokens: per gene n_freq frequent concretes, then one bucket holding the rest.
    # The LAST gene gets no tokens at all (like a gene absent from the training split).
    tok_gene, tok_kind, tok_conc = [], [], []
    conc_tok = np.full(NC, -1)
    pool_members = {}
    for g in range(G - 1):
        cs = np.arange(g * per_gene, (g + 1) * per_gene)
        for c in cs[:n_freq]:
            conc_tok[c] = len(tok_gene)
            tok_gene.append(g), tok_kind.append(0), tok_conc.append(c)
        b = len(tok_gene)
        tok_gene.append(g), tok_kind.append(1), tok_conc.append(-1)
        conc_tok[cs[n_freq:]] = b
        pool_members[b] = cs[n_freq:]
    tok_gene, tok_kind, tok_conc = map(np.array, (tok_gene, tok_kind, tok_conc))
    V = len(tok_gene)
    pool_ptr = np.zeros(V + 1, np.int64)
    pool_conc, pool_w = [], []
    for t in range(V):
        m = pool_members.get(t, np.array([], np.int64))
        pool_ptr[t + 1] = pool_ptr[t] + len(m)
        pool_conc += list(m)
        pool_w += list(rng.integers(1, 10, len(m)).astype(float))
    gene_tok_start = np.searchsorted(tok_gene, np.arange(G), 'left')
    gene_tok_end = np.searchsorted(tok_gene, np.arange(G), 'right')
    # patients (train patients never carry the token-less gene; val patients may)
    n_alt = rng.integers(1, 12, n_pat)
    split = (rng.random(n_pat) < 0.2).astype(np.int8)
    tr_pool = np.nonzero(conc_tok >= 0)[0]
    genes, concs = [], []
    for n, sp in zip(n_alt, split):
        c = np.sort(rng.choice(NC if sp == 1 else tr_pool, size=n, replace=False))
        concs.append(c)
        genes.append(conc_gene_idx[c])
    conc = np.concatenate(concs)
    gene = np.concatenate(genes)
    offsets = np.concatenate([[0], np.cumsum(n_alt)])
    var_tok = conc_tok[conc].copy()
    va_slot = np.repeat(split == 1, n_alt)
    drop = va_slot & (rng.random(len(conc)) < 0.1)            # some unmappable val alterations
    var_tok[drop] = -1
    vbin = rng.integers(0, 11, len(conc))
    disease = rng.integers(0, n_levels, n_pat)
    lvl_unl = np.zeros(n_levels, bool)
    lvl_unl[-1] = True
    tok_occ = rng.integers(1, 50, V).astype(float)
    vocab = dict(real_gene_ids=np.arange(G) + 5, levels=np.array([f'L{i}' for i in range(n_levels)]),
                 lvl_unlabeled=lvl_unl, conc_gene_idx=conc_gene_idx, tok_gene=tok_gene, tok_kind=tok_kind,
                 tok_conc=tok_conc, tok_occ=tok_occ, gene_tok_start=gene_tok_start, gene_tok_end=gene_tok_end,
                 pool_ptr=pool_ptr, pool_conc=np.array(pool_conc, np.int64), pool_w=np.array(pool_w),
                 vaf_rep=np.linspace(0, 1, 11).astype(np.float32), conc_tok=conc_tok)
    seq = dict(sample_ids=np.array([f's{i}' for i in range(n_pat)]), split=split, disease=disease,
               labeled=~lvl_unl[disease], offsets=offsets, gene_idx=gene.astype(np.int16),
               var_tok=var_tok.astype(np.int32), conc=conc.astype(np.int32), vaf_bin=vbin.astype(np.int8),
               aa_vaf=rng.random(len(conc)).astype(np.float32))
    emb = {'P_conc': torch.randn(NC, d), 'Pbar_tok': torch.randn(V, d)}
    return vocab, seq, emb


torch.manual_seed(0)
rng = np.random.default_rng(0)
dev = torch.device('cpu')
PASS = []


def ok(name):
    PASS.append(name)
    print(f'PASS {name}', flush=True)


vocab, seq, emb = make_toy(rng)
vocab['vaf_rep'] = np.concatenate([np.linspace(0, 1, 10), [0.0]]).astype(np.float32)   # real file: class 10 (missing) = 0.0
G = len(vocab['real_gene_ids'])
n_pat = len(seq['sample_ids'])
n_tot = len(seq['conc'])
timed = rng.random(n_tot) < 0.5
cbin = np.where(timed, rng.choice(N_BIN, n_tot, p=np.r_[np.full(17, 0.02), 0.06, 0.1, 0.1, 0.4][:N_BIN] / np.r_[np.full(17, 0.02), 0.06, 0.1, 0.1, 0.4][:N_BIN].sum()), -1)
cbin = cbin.astype(np.int8)
tmp = tempfile.mkdtemp(prefix='run9_ccf_selftest_')
np.savez(f'{tmp}/vocab.npz', **vocab)
np.savez(f'{tmp}/sequences.npz', **seq)
torch.save(emb, f'{tmp}/embeddings.pt')
np.savez(f'{tmp}/ccf_targets.npz', cbin=cbin, sample_ids=seq['sample_ids'])
np.savez(f'{tmp}/prefix_labels.npz', purity=rng.integers(0, 4, n_pat).astype(np.int8), purity_labels=np.array(list('abcd')),
         site=rng.integers(0, 5, n_pat).astype(np.int8), site_labels=np.array(list('vwxyz')), sample_ids=seq['sample_ids'])
D = CCFData(tmp, tmp)


def small(dropout=0.0):
    tl = torch.log(torch.as_tensor(vocab['tok_occ'])).float()
    return CCFDecoder(n_site=5, P_conc=emb['P_conc'][:, :64].contiguous(), Pbar_tok=emb['Pbar_tok'][:, :64].contiguous(),
                      gene_tok_start=vocab['gene_tok_start'], gene_tok_end=vocab['gene_tok_end'], tok_logprior=tl,
                      tok_kind=vocab['tok_kind'], tok_conc=vocab['tok_conc'], pool_ptr=vocab['pool_ptr'],
                      pool_conc=vocab['pool_conc'], pool_w=vocab['pool_w'], vaf_rep=vocab['vaf_rep'], n_genes=G,
                      n_levels=len(vocab['levels']), d=64, nhead=4, dff=64, nlayers=2, dropout=dropout,
                      conc_tok=vocab['conc_tok'], n_sig=4).to(dev)


# ----------------------------------------------------------------------------- C1 collate
o = seq['offsets']
ids = np.arange(80)
b = D.collate(ids, np.random.default_rng(5))
for r, i in enumerate(ids):
    a, e = o[i], o[i + 1]
    raw = [(int(seq['gene_idx'][k]), int(seq['var_tok'][k]), int(seq['conc'][k]), int(cbin[k])) for k in range(a, e)]
    n = int(b['n'][r])
    assert n == (e - a) + 1 and int(b['n_alt'][r]) == e - a
    got = list(zip(*(b[k][r, :n].tolist() for k in ('gene', 'tok', 'conc', 'cbin'))))
    nt = sum(1 for x in raw if x[3] >= 0)
    assert got[nt] == (G + 1, -1, 0, -1), 'separator position'
    tg, ug = got[:nt], got[nt + 1:]
    assert sorted(tg + ug) == sorted(raw), 'multiset'
    assert all(x[3] >= 0 for x in tg) and all(x[3] == -1 for x in ug)
    assert all(tg[k][3] >= tg[k + 1][3] for k in range(nt - 1)), 'descending bins'
    assert ug == [x for x in raw if x[3] < 0], 'untimed original order'
    assert (b['tok'][r, n:] == -1).all() and (b['cbin'][r, n:] == -1).all() and (b['gene'][r, n:] == 0).all()
assert torch.equal(b['sig'], torch.as_tensor(D.pur[ids])) and torch.equal(b['site'], torch.as_tensor(D.site[ids]))
b2 = D.collate(ids, np.random.default_rng(5))
assert all(torch.equal(b[k], b2[k]) for k in b)
diff = 0
for sd in range(6, 16):
    b3 = D.collate(ids, np.random.default_rng(sd))
    diff += int(any(not torch.equal(b[k][r], b3[k][r]) for r in range(len(ids)) for k in ('conc',)))
    assert torch.equal(b['cbin'], b3['cbin']), 'bin sequence must not depend on the tie order'
assert diff > 0
# uniformity: a patient with exactly one 3-way tie
cnt = {}
for i in range(n_pat):
    a, e = o[i], o[i + 1]
    c = cbin[a:e][cbin[a:e] >= 0]
    u, k = np.unique(c, return_counts=True)
    if len(c) == 3 and k.tolist() == [3] and len(set(seq['conc'][a:e][cbin[a:e] >= 0].tolist())) == 3:
        pid = i
        break
else:
    pid = None
if pid is not None:
    r_ = np.random.default_rng(99)
    for _ in range(1200):
        k = tuple(D.collate(np.array([pid]), r_)['conc'][0, :3].tolist())
        cnt[k] = cnt.get(k, 0) + 1
    assert len(cnt) == 6 and min(cnt.values()) > 140 and max(cnt.values()) < 260, cnt
ok('C1 collate == independent construction; tie order seeded, bin sequence independent of it' + ('; 3-way tie uniform over 6 orders' if pid is not None else ''))


# ----------------------------------------------------------------------------- C2 brute force
def brute(m, batch):
    tot = {k: 0.0 for k in ('lg', 'lv', 'lc', 'ag', 'av', 'ac')}
    cnt_ = {'ng': 0, 'nv': 0, 'nc': 0}
    for r in range(batch['gene'].shape[0]):
        n = int(batch['n'][r])
        gene, tok, conc, cb = (batch[k][r, :n] for k in ('gene', 'tok', 'conc', 'cbin'))
        x = [m.prefix_embed(batch['dis'][r:r + 1], batch['sig'][r:r + 1], batch['site'][r:r + 1])[0]]
        for k in range(n):
            x.append(m.sep_emb if int(gene[k]) == G + 1 else m.alt_embed_ccf(conc[k:k + 1], cb[k:k + 1])[0])
        x = torch.stack(x)[None]
        segc, prev, prevbin = 0, -1, N_BIN - 1
        for p in range(n + 1):
            h = m.encode(x[:, :p + 1])[0, p]
            lg = m.gene_head(h).clone()
            for c in range(G + 2):                                              # independent grammar
                if (c < G and segc == 1 and c < prev) or (c == G and segc != 1) or (c == G + 1 and segc != 0):
                    lg[c] = float('-inf')
            if p < n:
                tgt = int(gene[p])
            else:
                if int(batch['n_alt'][r]) >= MAX_ALTS:
                    break
                tgt = G
            tot['lg'] += float(-F.log_softmax(lg, -1)[tgt]); cnt_['ng'] += 1; tot['ag'] += int(lg.argmax() == tgt)
            if p == n:
                break
            if tgt == G + 1:
                segc += 1; prev = -1
                continue
            if segc == 0:
                assert int(cb[p]) >= 0 and int(cb[p]) <= prevbin
                cl = m.ccf_head(torch.cat([h, m.gene_emb(gene[p:p + 1])[0], m.P_conc[conc[p]]]))
                cl[torch.arange(N_BIN) > prevbin] = float('-inf')
                tot['lc'] += float(-F.log_softmax(cl, -1)[cb[p]]); cnt_['nc'] += 1; tot['ac'] += int(cl.argmax() == cb[p])
                prevbin = int(cb[p])
            else:
                assert int(cb[p]) == -1
            if int(tok[p]) >= 0:
                q = m.variant_query(h[None], gene[p:p + 1])
                lv = m.variant_logits(q, tgt)[0]
                loc = int(tok[p]) - m._gs[tgt]
                tot['lv'] += float(-F.log_softmax(lv, -1)[loc]); cnt_['nv'] += 1; tot['av'] += int(lv.argmax() == loc)
            prev = tgt
    return tot, cnt_


m = small().eval()
with torch.no_grad():
    bb = D.collate(np.arange(24), np.random.default_rng(1))
    bb['dis'] = torch.as_tensor(D.dis_in[np.arange(24)])
    out = m(bb)
    tot, cn = brute(m, bb)
assert int(out['n_gene']) == cn['ng'] and int(out['n_var']) == cn['nv'] and int(out['n_ccf']) == cn['nc']
for k, kk, nn_ in (('loss_gene', 'lg', 'ng'), ('loss_var', 'lv', 'nv'), ('loss_ccf', 'lc', 'nc')):
    assert abs(float(out[k]) - tot[kk] / cn[nn_]) < 1e-4, (k, float(out[k]), tot[kk] / cn[nn_])
assert int(out['acc_gene']) == tot['ag'] and int(out['acc_var']) == tot['av'] and int(out['acc_ccf']) == tot['ac']
n_timed_expected = int((np.concatenate([cbin[o[i]:o[i + 1]] for i in range(24)]) >= 0).sum())
assert int(out['n_ccf']) == n_timed_expected, 'ccf loss counts exactly the timed alterations'
ok(f'C2 forward == brute-force loop (independent grammar); CCF loss on exactly the {n_timed_expected} timed alterations')

# ----------------------------------------------------------------------------- C3 causality
with torch.no_grad():
    G_ = bb['gene']
    is_sep = G_ == G + 1
    xi = torch.where(is_sep[..., None], m.sep_emb.expand(*G_.shape, -1), m.alt_embed_ccf(bb['conc'], bb['cbin']))
    x = torch.cat([m.prefix_embed(bb['dis'], bb['sig'], bb['site'])[:, None], xi], 1)
    hf = m.encode(x)
    for t in (0, 2, 5):
        assert (hf[:, :t + 1] - m.encode(x[:, :t + 1])).abs().max() < 1e-5
    x2 = x.clone(); x2[:, 4] += 1.0
    h2 = m.encode(x2)
    assert (h2[:, :4] - hf[:, :4]).abs().max() < 1e-6 and (h2[:, 4:] - hf[:, 4:]).abs().max() > 1e-4
ok('C3 causality')


# ----------------------------------------------------------------------------- C4 greedy == teacher-forced argmax
def to_batch(go, dis, sig, site):
    rows = []
    for r in range(len(dis)):
        n = int(go['n'][r])
        sg = go['seg'][r, :n].tolist()
        f = lambda k: (int(go['gene'][r, k]), int(go['tok'][r, k]), int(go['conc'][r, k]), int(go['cbin'][r, k]))
        rows.append([f(k) for k in range(n) if sg[k] == 0] + [(G + 1, -1, 0, -1)] + [f(k) for k in range(n) if sg[k] == 1])
    T = max(len(x) for x in rows)
    bt = {k: torch.full((len(rows), T), fl, dtype=torch.long) for k, fl in (('gene', 0), ('tok', -1), ('conc', 0), ('cbin', -1))}
    for r, it in enumerate(rows):
        for k, (g_, t_, c_, v_) in enumerate(it):
            bt['gene'][r, k], bt['tok'][r, k], bt['conc'][r, k], bt['cbin'][r, k] = g_, t_, c_, v_
    bt['n'] = torch.tensor([len(x) for x in rows])
    bt['n_alt'] = go['n'].cpu()
    bt['dis'], bt['sig'], bt['site'] = dis.cpu(), sig.cpu(), site.cpu()
    return bt


n_seen = {'timed': 0, 'untimed': 0, 'rows': 0}
for seed_, sep_bias in ((1, 0.0), (2, 0.5), (3, 1.0), (4, 2.0), (5, 0.3), (6, 0.7)):
    torch.manual_seed(seed_)
    m = small().eval()
    with torch.no_grad():                                                # END and |U more likely so that greedy patients finish
        m.gene_head.bias[G] += 1.0
        m.gene_head.bias[G + 1] += sep_bias
    dis = torch.randint(0, len(vocab['levels']) + 1, (32,))
    sig = torch.randint(0, 4, (32,))
    site = torch.randint(0, 6, (32,))
    go = m.generate(dis, temperature=0, max_alts=20, sig=sig, site=site)
    keep = ~go['truncated']
    if keep.sum() == 0:
        continue
    go = {k: (v[keep] if torch.is_tensor(v) and v.shape[0] == 32 else v) for k, v in go.items()}
    dis, sig, site = dis[keep], sig[keep], site[keep]
    bt = to_batch(go, dis, sig, site)
    with torch.no_grad():
        G_ = bt['gene']
        is_sep = (torch.arange(G_.shape[1])[None] < bt['n'][:, None]) & (G_ == G + 1)
        xi = torch.where(is_sep[..., None], m.sep_emb.expand(*G_.shape, -1), m.alt_embed_ccf(bt['conc'], bt['cbin']))
        x = torch.cat([m.prefix_embed(bt['dis'], bt['sig'], bt['site'])[:, None], xi], 1)
        h = m.encode(x)
        for r in range(len(dis)):
            n = int(bt['n'][r]); segc, prev, prevbin = 0, -1, N_BIN - 1
            n_seen['rows'] += 1
            for p in range(n + 1):
                lg = m.gene_logits(h[r, p], torch.tensor(prev), torch.tensor(segc)).masked_fill(~m.gene_ok_ext, float('-inf'))
                tgt = int(G_[r, p]) if p < n else G
                assert int(lg.argmax()) == tgt, (seed_, r, p, int(lg.argmax()), tgt)
                if p == n:
                    break
                if tgt == G + 1:
                    segc += 1; prev = -1
                    continue
                q = m.variant_query(h[r, p][None], G_[r, p:p + 1])
                assert m._gs[tgt] + int(m.variant_logits(q, tgt)[0].argmax()) == int(bt['tok'][r, p])
                if segc == 0:
                    cl = m.ccf_logits(h[r, p][None], G_[r, p:p + 1], bt['conc'][r, p:p + 1], torch.tensor([prevbin]))[0]
                    assert int(cl.argmax()) == int(bt['cbin'][r, p])
                    prevbin = int(bt['cbin'][r, p]); n_seen['timed'] += 1
                else:
                    n_seen['untimed'] += 1
                prev = tgt
assert n_seen['timed'] > 20 and n_seen['untimed'] > 20 and n_seen['rows'] > 20, n_seen
ok(f"C4 greedy generate == teacher-forced argmax ({n_seen['rows']} patients, {n_seen['timed']} timed + {n_seen['untimed']} untimed steps, purity + site given)")

# ----------------------------------------------------------------------------- C5 sampled grammar + repeat rule
m = small().eval()
norep = np.ones(len(vocab['conc_gene_idx']), bool)
dis = torch.randint(0, len(vocab['levels']) + 1, (400,))
gs = m.generate(dis, temperature=1.0, max_alts=30, norep=norep, sig=torch.randint(0, 4, (400,)), site=torch.randint(0, 5, (400,)))
n_tr = 0
for r in range(400):
    n = int(gs['n'][r]); sg = gs['seg'][r, :n].numpy(); gg = gs['gene'][r, :n].numpy(); cc = gs['conc'][r, :n].numpy()
    tk = gs['tok'][r, :n].numpy(); cb = gs['cbin'][r, :n].numpy()
    assert (np.diff(sg) >= 0).all() and sg.min(initial=0) >= 0 and sg.max(initial=0) <= 1
    assert ((cb >= 0) == (sg == 0)).all() and (cb <= N_BIN - 1).all()
    assert (np.diff(cb[sg == 0]) <= 0).all(), 'descending bins'
    assert (np.diff(gg[sg == 1]) >= 0).all(), 'rank order in the untimed block'
    assert len(set(cc.tolist())) == n, 'repeated specific variant'
    assert (vocab['tok_gene'][tk] == gg).all() and m.gene_ok_ext[torch.as_tensor(gg)].all()
    assert (gs['gene'][r, n:] == -1).all() and (gs['cbin'][r, n:] == -1).all()
    n_tr += int(gs['truncated'][r])
assert (gs['n'][~gs['truncated']] <= 30).all() and (gs['n'][gs['truncated']] == 30).all()
assert set(gs['seg'][gs['seg'] >= 0].unique().tolist()) == {0, 1}
ok(f'C5 sampled grammar, no repeated specific variant ({n_tr}/400 truncated at 30)')

# ----------------------------------------------------------------------------- C6 capped patient
m = small().eval()
gene = np.sort(rng.integers(0, G - 1, MAX_ALTS))
c_ = np.array([np.nonzero(vocab['conc_gene_idx'] == g)[0][0] for g in gene])
tk = vocab['conc_tok'][c_]
nt = 60
bins = np.sort(rng.integers(0, N_BIN, nt))[::-1]
items = [(g, t, c, int(bins[i]) if i < nt else -1) for i, (g, t, c) in enumerate(zip(gene, tk, c_))]
items = items[:nt] + [(G + 1, -1, 0, -1)] + items[nt:]
bt = {k: torch.tensor([[it[i] for it in items]]) for i, k in enumerate(('gene', 'tok', 'conc', 'cbin'))}
bt.update(n=torch.tensor([MAX_ALTS + 1]), n_alt=torch.tensor([MAX_ALTS]), dis=torch.tensor([0]), sig=torch.tensor([0]), site=torch.tensor([0]))
with torch.no_grad():
    o_ = m(bt)
    assert int(o_['n_gene']) == MAX_ALTS + 1, int(o_['n_gene'])                # no END target
    bt['n_alt'] = torch.tensor([MAX_ALTS - 1])
    assert int(m(bt)['n_gene']) == MAX_ALTS + 2
ok('C6 capped patient: no END target; 129 positions fit max_len')

# ----------------------------------------------------------------------------- C7 prefix
m = small().train()
bb = D.collate(np.arange(16), np.random.default_rng(2))
m.eval()
with torch.no_grad():
    l0 = float(m(bb)['loss'])
    b_s = dict(bb); b_s['site'] = (bb['site'] + 1) % 5
    b_p = dict(bb); b_p['sig'] = (bb['sig'] + 1) % 4
    assert abs(float(m(b_s)['loss']) - l0) > 1e-6 and abs(float(m(b_p)['loss']) - l0) > 1e-6
    d_ = bb['dis']
    assert torch.equal(m.prefix_embed(d_, bb['sig'], None), m.prefix_embed(d_, bb['sig'], torch.full_like(d_, 5)))
    assert float(m.site_emb.weight[5].abs().max()) == 0.0 and float(m.sig_emb.weight[4].abs().max()) == 0.0
m.train()
m(bb)['loss'].backward()
assert m.site_emb.weight.grad is not None and float(m.site_emb.weight.grad[:5].abs().sum()) > 0 and m.sig_emb.weight.grad is not None
ok('C7 purity and site are used (loss changes, gradients flow); NO_SITE / NO_SIGNATURE rows are zero')

# ----------------------------------------------------------------------------- C8 train end-to-end + exact resume (CPU)
base = [sys.executable, f'{HERE}/train_ccf.py', '--data-dir', tmp, '--ccf-dir', tmp, '--epochs', '2', '--bs', '128', '--no-run8-init',
        '--save-every', '5', '--eval-bs', '256']
cmdA = base + ['--ckpt-dir', f'{tmp}/ckA']
rA = subprocess.run(cmdA, capture_output=True, text=True)
assert rA.returncode == 0, rA.stdout[-2000:] + rA.stderr[-2000:]
rows = open(f'{tmp}/ckA/metrics.csv').read().strip().splitlines()
assert len(rows) == 3, rows
vals = [float(x) for x in rows[-1].split(',')[:-1] if x]
assert np.isfinite(vals).all()
cmdB = base + ['--ckpt-dir', f'{tmp}/ckB']
r1 = subprocess.run(cmdB + ['--stop-after-steps', '23'], capture_output=True, text=True)
assert r1.returncode == 0 and 'checkpointed at epoch=1' in r1.stdout, r1.stdout[-2000:] + r1.stderr[-2000:]
r2 = subprocess.run(cmdB, capture_output=True, text=True)
assert r2.returncode == 0 and 'RESUMED' in r2.stdout and 'TRAINING DONE' in r2.stdout, r2.stdout[-2000:] + r2.stderr[-2000:]
sa = torch.load(f'{tmp}/ckA/last.pt', weights_only=False)['model']
sb = torch.load(f'{tmp}/ckB/last.pt', weights_only=False)['model']
assert all(torch.equal(sa[k], sb[k]) for k in sa), 'resumed run differs from the uninterrupted run'
import csv
va = [float(r['val_loss']) for r in csv.DictReader(open(f'{tmp}/ckA/metrics.csv'))]
vb = [float(r['val_loss']) for r in csv.DictReader(open(f'{tmp}/ckB/metrics.csv'))]
assert va == vb and len(va) == 2, (va, vb)
rep = ccf_bin_rep()
assert abs(rep[0] - 0.025) < 1e-7 and abs(rep[19] - 0.975) < 1e-7 and rep[20] == 1.0 and len(rep) == N_BIN
ok('C8 train_ccf.py runs (2 epochs, finite metrics); stop at step 23 + resume == uninterrupted run, weights bit-identical')

print(f'ALL {len(PASS)} PASSED')
