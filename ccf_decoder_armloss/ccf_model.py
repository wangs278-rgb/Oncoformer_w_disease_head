"""run9 CCFDecoder — patient decoder whose alterations are ordered by DESCENDING CCF (no route token, no latent z).

Sequence (teacher forcing), items = alterations + one separator:
    input  : [DISEASE + purity4 + site]  timed_1 .. timed_a  |U  untimed_1 .. untimed_b
    target :  timed_1 ..                 |U  untimed_1 ..        END
timed   = alterations with a CCF (classes 1-3, see ccf_prep.py / ccf_prep_armloss.py), sorted by CCF bin
          from the highest down (bin 20 = CCF >= 1, 0.05 bins); ties (same bin) in RANDOM order, redrawn by the data loader each batch.
untimed = everything else (copy-number events, short variants without a CCF), original frequency-rank order, no value.
Per alteration step the chain is the parent's:  gene (G genes, END = G, |U = G+1)  ->  variant token of that gene  ->  CCF bin (timed only).
Grammar, enforced in training AND sampling:
    segment 0 (timed):   any gene (no rank order), |U allowed, END not;   CCF bin <= the previous timed alteration's bin
    segment 1 (untimed): genes non-decreasing in frequency rank (as in the parent), END allowed, |U not.
Input embedding of an alteration = run8's frozen token embedding + run8's Fourier encoder at the bin centre of its CCF
(untimed: the parent's VAF-missing value 0.0).  Prefix = disease_emb + purity4 embedding (parent's sig_emb) + site embedding.
Transformer, heads, run8 initialisation: the parent's.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from decoder_model import PatientDecoder

MAX_ALTS = 127                      # run8 cap: patients at the cap get no END target
N_BIN = 21                          # CCF bins 0..20 (0.05 steps, bin 20 = CCF >= 1)
CCF_STEP = 0.05
VAF_NA = 10                         # parent's "VAF missing" class (representative value 0.0)


def ccf_bin_rep():
    """Representative CCF of each bin: its centre; the last bin is CCF >= 1 (capped at 1) -> 1.0."""
    r = (np.arange(N_BIN) + 0.5) * CCF_STEP
    r[-1] = 1.0
    return r.astype(np.float32)


class CCFDecoder(PatientDecoder):
    def __init__(self, n_site=5, **kw):
        kw.setdefault('max_len', MAX_ALTS + 2)              # [prefix] + 127 alterations + 1 separator
        emb_std = float(kw.get('emb_std', 1.0))
        super().__init__(**kw)
        G, d = self.n_genes, self.d
        self.SEP_U = G + 1
        old = self.gene_head
        self.gene_head = nn.Linear(d, G + 2)
        with torch.no_grad():                                # keep the parent's init for genes + END
            self.gene_head.weight[:G + 1].copy_(old.weight)
            self.gene_head.bias[:G + 1].copy_(old.bias)
        self.sep_emb = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.sep_emb, std=emb_std)
        self.gene_ok_ext = torch.cat([self.gene_ok_ext, torch.ones(1, dtype=torch.bool)])
        self.register_buffer('ccf_rep', torch.as_tensor(ccf_bin_rep()), persistent=False)
        self.ccf_head = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.LayerNorm(d), nn.Linear(d, N_BIN))
        self.n_site = n_site
        self.site_emb = nn.Embedding(n_site + 1, d)           # last row = NO_SITE (zero; unused when every patient has a site)
        nn.init.normal_(self.site_emb.weight, std=emb_std)
        with torch.no_grad():
            self.site_emb.weight[n_site].zero_()

    # ------------------------------------------------------------------ blocks
    def prefix_embed(self, dis, sig=None, site=None):
        x = super().prefix_embed(dis, sig)                    # disease (+ purity4 through the parent's signature input)
        if site is None:
            site = torch.full_like(dis, self.n_site)
        return x + self.site_emb(site)

    def alt_embed_ccf(self, conc, cbin):
        """cbin >= 0: Fourier(bin centre); cbin < 0 (untimed): Fourier(0.0) = the parent's VAF-missing embedding."""
        rep = torch.where(cbin >= 0, self.ccf_rep[cbin.clamp_min(0)], self.vaf_rep[VAF_NA].expand(cbin.shape))
        return self.P_conc[conc] + self.fourier(rep)

    def gene_logits(self, h, prev_gene, seg):
        """h [...,d]; prev_gene [...] (-1 = segment start); seg [...] 0 timed / 1 untimed.  fp32 head."""
        with torch.autocast(device_type=h.device.type, enabled=False):
            lg = self.gene_head(h.float())
        G = self.n_genes
        ar = torch.arange(G + 2, device=h.device)
        p, s = prev_gene.unsqueeze(-1), seg.unsqueeze(-1)
        bad = ((ar < p) & (ar < G) & (s == 1)) | ((ar == G) & (s != 1)) | ((ar == self.SEP_U) & (s != 0))
        return lg.masked_fill(bad, float('-inf'))

    def ccf_logits(self, h, g, conc, prev_bin):
        """Bins above the previous timed alteration's bin are impossible (descending order)."""
        with torch.autocast(device_type=h.device.type, enabled=False):
            lg = self.ccf_head(torch.cat([h.float(), self.gene_emb(g), self.P_conc[conc]], -1))
        ar = torch.arange(N_BIN, device=h.device)
        return lg.masked_fill(ar[None] > prev_bin[:, None], float('-inf'))

    # ------------------------------------------------------------------ training forward
    def forward(self, batch):
        """batch (CCFData.collate): gene [B,T] holds genes and the separator (G+1); tok -1 at the separator; cbin [B,T] (-1 untimed /
        separator / pad); n [B] = items (alterations + 1); n_alt [B]; dis, sig (purity4), site [B]."""
        dis, gene, tok, conc, cbin, n, n_alt = (batch[k] for k in ('dis', 'gene', 'tok', 'conc', 'cbin', 'n', 'n_alt'))
        sig = batch['sig'] if self.n_sig > 0 else None
        site = batch['site']
        B, T = gene.shape
        dev = gene.device
        j = torch.arange(T, device=dev)[None]
        valid = j < n[:, None]
        is_sep = valid & (gene == self.SEP_U)
        alt_valid = valid & ~is_sep
        x_alt = self.alt_embed_ccf(conc.clamp_min(0), cbin)
        x_items = torch.where(is_sep[..., None], self.sep_emb.to(x_alt.dtype).expand_as(x_alt), x_alt)
        x = torch.cat([self.prefix_embed(dis, sig, site)[:, None].to(x_items.dtype), x_items], 1)
        jj = torch.arange(T + 1, device=dev)[None]
        h = self.encode(x, jj > n[:, None])

        # ---- gene / separator / END
        tgt_gene = torch.full((B, T + 1), -100, dtype=torch.long, device=dev)
        tgt_gene[:, :T] = torch.where(valid, gene, tgt_gene[:, :T])
        tgt_gene.scatter_(1, n[:, None], self.END)
        capped = n_alt >= MAX_ALTS
        if capped.any():
            tgt_gene[capped, n[capped]] = -100          # END of a capped patient is run8's truncation, not biology
        prev = torch.full((B, T + 1), -1, dtype=torch.long, device=dev)
        prev[:, 1:] = torch.where(alt_valid, gene, torch.full_like(gene, -1))
        seg = torch.zeros((B, T + 1), dtype=torch.long, device=dev)
        seg[:, 1:] = torch.cumsum(is_sep.long(), 1)
        glog = self.gene_logits(h, prev, seg)
        sel = tgt_gene >= 0
        loss_gene = F.cross_entropy(glog[sel], tgt_gene[sel])
        acc_gene = (glog[sel].argmax(-1) == tgt_gene[sel]).float().sum()
        n_gene = sel.sum()

        # ---- variant token (every alteration) and CCF bin (timed alterations)
        hb, hj = torch.nonzero(alt_valid, as_tuple=True)
        h_s = h[hb, hj]
        g_s, t_s, c_s, b_s = gene[hb, hj], tok[hb, hj], conc[hb, hj], cbin[hb, hj]
        tm = b_s >= 0
        pb = torch.full((B, T), N_BIN - 1, dtype=torch.long, device=dev)       # previous timed bin (first timed alteration: no limit)
        pb[:, 1:] = torch.where(cbin[:, :-1] >= 0, cbin[:, :-1], torch.full_like(cbin[:, :-1], N_BIN - 1))
        clog = self.ccf_logits(h_s[tm], g_s[tm], c_s[tm], pb[hb, hj][tm])
        loss_ccf = F.cross_entropy(clog, b_s[tm])
        acc_ccf = (clog.argmax(-1) == b_s[tm]).float().sum()
        n_ccf = tm.sum()

        m = t_s >= 0
        q = self.variant_query(h_s[m], g_s[m])
        g_m, t_m = g_s[m], t_s[m]
        order = torch.argsort(g_m)
        uniq, counts = torch.unique_consecutive(g_m[order], return_counts=True)
        loss_var_sum = torch.zeros((), device=dev)
        acc_var = torch.zeros((), device=dev)
        ptr = 0
        for g, cnt in zip(uniq.tolist(), counts.tolist()):
            idx = order[ptr:ptr + cnt]
            ptr += cnt
            lg = self.variant_logits(q[idx], g)
            local = t_m[idx] - self._gs[g]
            loss_var_sum = loss_var_sum + F.cross_entropy(lg, local, reduction='sum')
            acc_var = acc_var + (lg.argmax(-1) == local).float().sum()
        n_var = m.sum()
        loss_var = loss_var_sum / n_var.clamp_min(1)
        return {'loss': loss_gene + loss_var + loss_ccf, 'loss_gene': loss_gene, 'loss_var': loss_var, 'loss_ccf': loss_ccf,
                'acc_gene': acc_gene, 'n_gene': n_gene, 'acc_var': acc_var, 'n_var': n_var, 'acc_ccf': acc_ccf, 'n_ccf': n_ccf}

    # ------------------------------------------------------------------ sampling
    @torch.no_grad()
    def generate(self, dis, temperature=1.0, max_alts=MAX_ALTS, norep=None, generator=None, sig=None, site=None):
        """Returns dict of [B,max_alts] long tensors gene/tok/conc/cbin/seg (-1 padded; alterations only, in generation order =
        descending CCF bin, then the untimed block), n [B] alterations, truncated [B] (hit max_alts before END).
        cbin is -1 for untimed alterations.  norep: bool [n_concrete]; a flagged (specific) concrete appears at most once per patient.
        Genes can recur anywhere in the timed block, so the rule checks every gene the patient already carries (as BranchDecoder)."""
        assert not self.training, 'call model.eval() before generate()'
        if norep is not None:
            assert self._np['conc_tok'] is not None
            norep = np.asarray(norep.cpu() if torch.is_tensor(norep) else norep, dtype=bool)
        B, dev, G = dis.shape[0], dis.device, self.n_genes
        L = max_alts + 2
        X = torch.zeros(B, L, self.d, device=dev)
        X[:, 0] = self.prefix_embed(dis, sig, site)
        out = {k: torch.full((B, max_alts), -1, dtype=torch.long, device=dev) for k in ('gene', 'tok', 'conc', 'cbin', 'seg')}
        n_out = torch.zeros(B, dtype=torch.long, device=dev)
        alive = torch.ones(B, dtype=torch.bool, device=dev)
        prev = torch.full((B,), -1, dtype=torch.long, device=dev)
        prev_bin = torch.full((B,), N_BIN - 1, dtype=torch.long, device=dev)
        seg = torch.zeros(B, dtype=torch.long, device=dev)
        truncated = torch.zeros(B, dtype=torch.bool, device=dev)
        if norep is not None:
            h_gene = np.full((B, max_alts), -1, np.int64)
            h_conc = np.full((B, max_alts), -1, np.int64)
            h_n = np.zeros(B, np.int64)
            gcnt = np.zeros((B, G), np.int64)
            ntok = np.asarray(self._ge) - np.asarray(self._gs)

            def used(r, g):
                k = h_n[r]
                return h_conc[r, :k][h_gene[r, :k] == g]
        for t in range(L):
            cap = alive & (n_out >= max_alts)
            truncated |= cap
            alive &= ~cap
            if not alive.any():
                break
            h = self.encode(X[:, :t + 1])[:, t]
            glog = self.gene_logits(h, prev, seg).masked_fill(~self.gene_ok_ext, float('-inf'))
            if norep is not None:
                al = alive.cpu().numpy()
                rr, gg_ = np.nonzero((gcnt >= ntok[None]) & (gcnt > 0) & al[:, None])   # #invalid <= #used in gene
                blocked = [(r, g) for r, g in zip(rr, gg_) if not self._valid_tokens(used(r, g), g, norep)[0].any()]
                if blocked:
                    bt = torch.as_tensor(blocked, device=dev)
                    glog[bt[:, 0], bt[:, 1]] = float('-inf')
            g = self._sample(glog, temperature, generator)
            stop = alive & (g == self.END)
            alive = alive & ~stop
            sep = alive & (g == self.SEP_U)
            if sep.any():
                X[sep, t + 1] = self.sep_emb
                seg[sep] += 1
                prev[sep] = -1
            isalt = alive & (g < G)
            if not isalt.any():
                continue
            rows = torch.nonzero(isalt, as_tuple=True)[0]
            gr, hr = g[rows], h[rows]
            q = self.variant_query(hr, gr)
            tok = torch.empty_like(gr)
            order = torch.argsort(gr)
            uniq, counts = torch.unique_consecutive(gr[order], return_counts=True)
            if norep is not None:
                rows_np, gr_np, order_np = rows.cpu().numpy(), gr.cpu().numpy(), order.cpu().numpy()
            ptr = 0
            for gg, cnt in zip(uniq.tolist(), counts.tolist()):
                idx = order[ptr:ptr + cnt]
                lg = self.variant_logits(q[idx], gg)
                if norep is not None:
                    for li, i in enumerate(order_np[ptr:ptr + cnt]):
                        r = rows_np[i]
                        if gcnt[r, gg] == 0:
                            continue
                        valid, _ = self._valid_tokens(used(r, gg), gg, norep)
                        if not valid.all():
                            lg[li, torch.as_tensor(~valid, device=dev)] = float('-inf')
                ptr += cnt
                tok[idx] = self._gs[gg] + self._sample(lg, temperature, generator)
            conc = torch.where(self.tok_kind[tok] == 0, self.tok_conc[tok], torch.zeros_like(tok))
            bk = self.tok_kind[tok] == 1
            if bk.any():
                conc[bk] = self._draw_pool(tok[bk], generator)
            if norep is not None and bk.any():
                bk_np, conc_np, tok_np = bk.cpu().numpy(), conc.cpu().numpy(), tok.cpu().numpy()
                for i in np.nonzero(bk_np & (gcnt[rows_np, gr_np] > 0))[0]:
                    r = rows_np[i]
                    _, used_nr = self._valid_tokens(used(r, gr_np[i]), gr_np[i], norep)
                    if int(conc_np[i]) not in used_nr:
                        continue
                    b = int(tok_np[i])
                    pp = self._np['pool_ptr']
                    mem = self._np['pool_conc'][pp[b]:pp[b + 1]]
                    w = self._np['pool_w'][pp[b]:pp[b + 1]] * np.array([int(m_) not in used_nr for m_ in mem])
                    cs = np.cumsum(w)
                    if not cs[-1] > 0:
                        raise RuntimeError(f'exhausted bucket {b} was sampled')
                    u = float(torch.rand(1, device=dev, dtype=torch.float64, generator=generator)) * cs[-1]
                    conc[i] = int(mem[min(np.searchsorted(cs, u, side='right'), len(mem) - 1)])
            cb = torch.full_like(gr, -1)
            tr = seg[rows] == 0                                           # timed block: sample the CCF bin, <= previous bin
            if tr.any():
                lgc = self.ccf_logits(hr[tr], gr[tr], conc[tr], prev_bin[rows[tr]])
                cb[tr] = self._sample(lgc, temperature, generator)
                prev_bin[rows[tr]] = cb[tr]
            k = n_out[rows]
            out['gene'][rows, k], out['tok'][rows, k], out['conc'][rows, k] = gr, tok, conc
            out['cbin'][rows, k], out['seg'][rows, k] = cb, seg[rows]
            n_out[rows] += 1
            X[rows, t + 1] = self.alt_embed_ccf(conc, cb)
            prev[rows] = gr
            if norep is not None:
                kk = k.cpu().numpy()
                h_gene[rows_np, kk] = gr_np
                h_conc[rows_np, kk] = conc.cpu().numpy()
                h_n[rows_np] += 1
                np.add.at(gcnt, (rows_np, gr_np), 1)
        truncated |= alive
        out['n'] = n_out
        out['truncated'] = truncated
        return out


def build_ccf_decoder(vocab, emb, init=None, dropout=0.1, n_sig=4, n_site=5):
    """CCFDecoder from prep_data.py outputs (arguments as decoder_model.build_decoder; n_sig = purity classes, n_site = site classes)."""
    tok_logprior = torch.log(torch.as_tensor(vocab['tok_occ'], dtype=torch.float64)).float()
    lc = (init or {}).get('layer_cfg', {'d_model': 512, 'nhead': 8, 'dim_feedforward': 512,
                                         'num_layers': 4, 'layer_norm_eps': 1e-5})
    model = CCFDecoder(
        n_site=n_site,
        P_conc=emb['P_conc'], Pbar_tok=emb['Pbar_tok'], gene_tok_start=vocab['gene_tok_start'],
        gene_tok_end=vocab['gene_tok_end'], tok_logprior=tok_logprior, tok_kind=vocab['tok_kind'],
        tok_conc=vocab['tok_conc'], pool_ptr=vocab['pool_ptr'], pool_conc=vocab['pool_conc'],
        pool_w=vocab['pool_w'], vaf_rep=vocab['vaf_rep'], n_genes=len(vocab['real_gene_ids']),
        n_levels=len(vocab['levels']), d=lc['d_model'], nhead=lc['nhead'], dff=lc['dim_feedforward'],
        nlayers=lc['num_layers'], dropout=dropout, layer_norm_eps=lc['layer_norm_eps'],
        emb_std=float((init or {}).get('p_std', 1.0)), conc_tok=vocab.get('conc_tok'), n_sig=n_sig)
    if init is not None:
        model.load_run8_init(init)
    return model
