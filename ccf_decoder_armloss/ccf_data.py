"""Loading + batching for the CCFDecoder.  Reads run9's data/sequences.npz + vocab.npz (plain layout, as run9 code/seqdata.py:SeqData)
and adds ccf_targets.npz / prefix_labels.npz (ccf_prep.py, then ccf_prep_armloss.py).  The sequence order is built HERE, per batch:
    timed alterations (cbin >= 0) sorted by cbin from high to low, ties in random order drawn from the rng passed to collate();
    then the separator |U (gene code G+1);
    then the untimed alterations in their original (frequency-rank) order.
"""
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CCF_DIR = os.environ.get('CCF_DATA', os.path.join(HERE, 'data'))         # ccf_targets.npz + prefix_labels.npz (ccf_prep_armloss.py --out)


def env_dir(name, value=None):
    """value if given, else environment variable `name`; error if neither is set."""
    v = value or os.environ.get(name)
    if not v:
        raise SystemExit(f'set ${name} or pass the matching --option (see README.md)')
    return v


class CCFData:
    def __init__(self, data_dir, ccf_dir=CCF_DIR, targets='ccf_targets.npz', labels='prefix_labels.npz'):
        seq = np.load(f'{data_dir}/sequences.npz')                          # run9 plain layout (fields as SeqData)
        self.vocab = dict(np.load(f'{data_dir}/vocab.npz'))
        self.offsets = seq['offsets'].astype(np.int64)
        self.n_alt = np.diff(self.offsets)
        self.gene = seq['gene_idx'].astype(np.int64)
        self.tok = seq['var_tok'].astype(np.int64)
        self.conc = seq['conc'].astype(np.int64)
        self.split = seq['split']
        self.disease = seq['disease'].astype(np.int64)
        self.labeled = seq['labeled'].astype(bool)
        self.sample_ids = seq['sample_ids']
        self.dis_in = np.where(self.labeled, self.disease, len(self.vocab['levels']))   # decoder input token (NO_DISEASE if unlabelled)
        t = np.load(f'{ccf_dir}/{targets}')
        assert np.array_equal(t['sample_ids'], self.sample_ids), 'ccf targets not aligned with sequences.npz'
        self.cbin = t['cbin'].astype(np.int64)
        assert len(self.cbin) == len(self.gene) and self.cbin.min() >= -1 and self.cbin.max() <= 20
        L = np.load(f'{ccf_dir}/{labels}')
        assert np.array_equal(L['sample_ids'], self.sample_ids), 'prefix labels not aligned with sequences.npz'
        self.pur, self.site = L['purity'].astype(np.int64), L['site'].astype(np.int64)
        self.pur_labels, self.site_labels = L['purity_labels'], L['site_labels']
        self.G = len(self.vocab['real_gene_ids'])
        assert 0 <= self.pur.min() and self.pur.max() < len(self.pur_labels) and 0 <= self.site.min() and self.site.max() < len(self.site_labels)

    def collate(self, ids, rng):
        """Padded batch in CCFDecoder order.  rng: numpy Generator for the tie order (same rng state -> same batch)."""
        ids = np.asarray(ids)
        o, G, nb = self.offsets, self.G, len(ids)
        n_alt = o[ids + 1] - o[ids]
        rows = np.repeat(np.arange(nb), n_alt)
        flat = np.concatenate([np.arange(o[i], o[i + 1]) for i in ids])
        within = flat - np.repeat(o[ids], n_alt)
        cb = self.cbin[flat]
        timed = cb >= 0
        k1 = np.where(timed, 0, 2)                                           # block: timed 0, separator 1, untimed 2
        k2 = np.where(timed, -cb, 0)                                         # timed: highest bin first
        k3 = np.where(timed, rng.random(len(flat)), within.astype(np.float64))   # ties: random; untimed: original order
        R = np.concatenate([rows, np.arange(nb)])
        K1 = np.concatenate([k1, np.ones(nb, np.int64)])
        K2 = np.concatenate([k2, np.zeros(nb, np.int64)])
        K3 = np.concatenate([k3, np.zeros(nb)])
        SRC = np.concatenate([flat, np.full(nb, -1)])
        od = np.lexsort((K3, K2, K1, R))
        R, SRC = R[od], SRC[od]
        n = n_alt + 1
        start = np.concatenate([[0], np.cumsum(n)[:-1]])
        pos = np.arange(len(R)) - start[R]
        T = int(n.max())
        is_alt = SRC >= 0
        out = {}
        for name, src, fill in (('gene', self.gene, 0), ('tok', self.tok, -1), ('conc', self.conc, 0), ('cbin', self.cbin, -1)):
            a = np.full((nb, T), fill, dtype=np.int64)
            a[R[is_alt], pos[is_alt]] = src[SRC[is_alt]]
            out[name] = a
        out['gene'][R[~is_alt], pos[~is_alt]] = G + 1
        out = {k: torch.from_numpy(v) for k, v in out.items()}
        out['n'] = torch.from_numpy(n.astype(np.int64))
        out['n_alt'] = torch.from_numpy(n_alt.astype(np.int64))
        out['dis'] = torch.from_numpy(self.dis_in[ids])
        out['sig'] = torch.from_numpy(self.pur[ids])
        out['site'] = torch.from_numpy(self.site[ids])
        return out
