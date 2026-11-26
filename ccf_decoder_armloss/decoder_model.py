"""run9 PatientDecoder — disease-conditioned autoregressive generator of patients (no latent z).
Parent class of ccf_model.CCFDecoder; copied unchanged from run9 code/decoder_model.py (only this line added).

Sequence (teacher forcing):
    input  : [DISEASE]   alt_1     alt_2   ...  alt_n
    target :  alt_1      alt_2     alt_3   ...  END
Each alteration step is predicted in a chain:  gene (G genes + END)  ->  variant token of that
gene  ->  VAF class (10 relative-VAF bins + NA).

Inputs of fed-back alterations use run8's FROZEN token embedding (ESM protein + ESM mutation ->
512, precomputed as P_conc) plus run8's Fourier VAF encoder at the VAF bin's representative value.
The transformer is causal and initialised from run8's 4 encoder layers.

Loss = CE(gene) + CE(variant | gene) + CE(VAF | gene, variant), each averaged over its steps.
Genes are generated in non-decreasing frequency-rank order (the data's canonical order); genes
ranked before the previous alteration's gene are masked in BOTH training and sampling.

Optional signature conditioning (n_sig > 0): an FMI signature embedding is ADDED to the [DISEASE] input token
(same sequence length, no extra loss). Its NO_SIGNATURE row starts at zero, so an untrained signature model
equals the disease-only model. n_sig = 0 (default) = the original model, unchanged.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierFractionEncoder(nn.Module):
    """Verbatim copy of Oncoformer_moco/oncoformer/models.py:FourierFractionEncoder (run8's aa_vaf
    encoder) so run8's weights load 1:1."""

    def __init__(self, embedding_dim=128, fourier_dim=32, f_min=1.0, f_max=64.0, add_logit=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.register_buffer("freqs", torch.logspace(math.log10(f_min), math.log10(f_max), fourier_dim))
        self.add_logit = add_logit
        in_dim = 2 * fourier_dim + 1 + (1 if add_logit else 0)
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, embedding_dim))

    def forward(self, x):
        x = x.clamp(0, 1)
        phases = 2 * math.pi * x.unsqueeze(-1) * self.freqs
        fe = torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)
        parts = [fe, x.unsqueeze(-1)]
        if self.add_logit:
            eps = 1e-5
            parts.append(torch.log((x + eps) / (1 - x + eps)).unsqueeze(-1))
        return self.proj(torch.cat(parts, dim=-1))


def _mlp(d_in, d):
    return nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.LayerNorm(d), nn.Linear(d, d))


class PatientDecoder(nn.Module):
    def __init__(self, *, P_conc, Pbar_tok, gene_tok_start, gene_tok_end, tok_logprior, tok_kind, tok_conc,
                 pool_ptr, pool_conc, pool_w, vaf_rep, n_genes, n_levels, n_vaf=11, d=512, nhead=8, dff=512,
                 nlayers=4, dropout=0.1, max_len=128, layer_norm_eps=1e-5, emb_std=1.0, conc_tok=None, n_sig=0):
        super().__init__()
        V = Pbar_tok.shape[0]
        self.n_genes, self.END, self.n_vaf, self.max_len, self.d = n_genes, n_genes, n_vaf, max_len, d
        self.NO_DISEASE = n_levels
        # frozen run8-derived tables (buffers: saved with the model, never trained)
        self.register_buffer('P_conc', P_conc.float(), persistent=False)
        self.register_buffer('Pbar_tok', Pbar_tok.float(), persistent=False)
        self.register_buffer('g_start', torch.as_tensor(gene_tok_start, dtype=torch.long), persistent=False)
        self.register_buffer('g_end', torch.as_tensor(gene_tok_end, dtype=torch.long), persistent=False)
        self._gs = [int(v) for v in gene_tok_start]                      # host copies: no GPU sync per gene
        self._ge = [int(v) for v in gene_tok_end]
        # genes with >= 1 variant token (+ END): only these can be SAMPLED (a gene absent from train has none)
        ok = torch.as_tensor(gene_tok_end, dtype=torch.long) > torch.as_tensor(gene_tok_start, dtype=torch.long)
        self.register_buffer('gene_ok_ext', torch.cat([ok, torch.ones(1, dtype=torch.bool)]), persistent=False)
        self.register_buffer('tok_kind', torch.as_tensor(tok_kind, dtype=torch.long), persistent=False)
        self.register_buffer('tok_conc', torch.as_tensor(tok_conc, dtype=torch.long), persistent=False)
        self.register_buffer('pool_ptr', torch.as_tensor(pool_ptr, dtype=torch.long), persistent=False)
        self.register_buffer('pool_conc', torch.as_tensor(pool_conc, dtype=torch.long), persistent=False)
        pw = torch.as_tensor(pool_w, dtype=torch.float64)
        self.register_buffer('pool_cum', torch.cumsum(pw, 0), persistent=False)          # float64 CDF
        self.register_buffer('vaf_rep', torch.as_tensor(vaf_rep, dtype=torch.float32), persistent=False)
        # host (numpy) copies used only by the sampling repeat rule (generate(norep=...))
        self._np = {'tok_kind': np.asarray(tok_kind, np.int64), 'tok_conc': np.asarray(tok_conc, np.int64),
                    'pool_ptr': np.asarray(pool_ptr, np.int64), 'pool_conc': np.asarray(pool_conc, np.int64),
                    'pool_w': np.asarray(pool_w, np.float64),
                    'conc_tok': None if conc_tok is None else np.asarray(conc_tok, np.int64)}

        # inputs
        self.disease_emb = nn.Embedding(n_levels + 1, d)            # last row = NO_DISEASE
        nn.init.normal_(self.disease_emb.weight, std=emb_std)
        self.pos_emb = nn.Embedding(max_len, d)
        nn.init.zeros_(self.pos_emb.weight)                           # keeps run8 init exact at step 0
        self.fourier = FourierFractionEncoder(d, 32, 1.0, 64.0, True)
        # causal transformer (same layer type/hyper-params as run8's encoder)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(
            d, nhead, dff, dropout=dropout, activation='gelu', batch_first=True, norm_first=False,
            layer_norm_eps=layer_norm_eps) for _ in range(nlayers)])
        # heads
        self.gene_head = nn.Linear(d, n_genes + 1)
        self.gene_emb = nn.Embedding(n_genes, d)
        nn.init.normal_(self.gene_emb.weight, std=0.02)
        self.var_q = _mlp(2 * d, d)
        self.var_out = nn.Linear(d, d)
        self.var_res = nn.Parameter(torch.zeros(V, d))
        self.var_bias = nn.Parameter(torch.as_tensor(tok_logprior, dtype=torch.float32).clone())
        self.vaf_head = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.LayerNorm(d), nn.Linear(d, n_vaf))
        # optional FMI signature input — created LAST so every other parameter initialises exactly as in the
        # disease-only model for the same seed
        self.n_sig = n_sig
        self.NO_SIGNATURE = n_sig
        if n_sig > 0:
            self.sig_emb = nn.Embedding(n_sig + 1, d)                 # last row = NO_SIGNATURE
            nn.init.normal_(self.sig_emb.weight, std=emb_std)
            with torch.no_grad():
                self.sig_emb.weight[n_sig].zero_()                    # unlabelled patients start disease-only

    # ------------------------------------------------------------------ init from run8
    def load_run8_init(self, init):
        for lyr, sd in zip(self.layers, init['layers']):
            lyr.load_state_dict(sd)
        fc = init['fourier_cfg']
        assert fc['fourier_dim'] == 32 and fc['add_logit'] and abs(fc['f_min'] - 1.0) < 1e-6 and abs(fc['f_max'] - 64.0) < 1e-3
        self.fourier.load_state_dict(init['fourier'])
        with torch.no_grad():
            self.gene_head.weight[:self.n_genes].copy_(init['gene_head_w'])
            self.gene_head.bias[:self.n_genes].copy_(init['gene_head_b'])

    # ------------------------------------------------------------------ building blocks
    def prefix_embed(self, dis, sig=None):
        """Input embedding of the conditioning (first) token: disease, plus the FMI signature if n_sig > 0
        (sig None -> NO_SIGNATURE). n_sig = 0: exactly disease_emb(dis); passing sig is an error."""
        x = self.disease_emb(dis)
        if self.n_sig == 0:
            if sig is not None:
                raise ValueError('signature given to a model built without signature conditioning (n_sig=0)')
            return x
        if sig is None:
            sig = torch.full_like(dis, self.NO_SIGNATURE)
        return x + self.sig_emb(sig)

    def alt_embed(self, conc, vbin):
        """Input embedding of alterations: frozen run8 token embedding + run8 Fourier(VAF rep)."""
        return self.P_conc[conc] + self.fourier(self.vaf_rep[vbin])

    @staticmethod
    def causal_mask(T, device):
        return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)   # True = blocked

    def encode(self, x, key_pad=None):
        """x: [B,T,d] input embeddings (without positions). Returns causal hidden states [B,T,d]."""
        T = x.shape[1]
        h = x + self.pos_emb.weight[:T][None]
        m = self.causal_mask(T, x.device)
        for lyr in self.layers:
            h = lyr(h, src_mask=m, src_key_padding_mask=key_pad)
        return h

    def gene_logits(self, h, prev_gene):
        """h [...,d]; prev_gene [...] (-1 = no constraint). Genes ranked before prev are masked.
        Output heads run in fp32 (autocast off) for logit precision."""
        with torch.autocast(device_type=h.device.type, enabled=False):
            lg = self.gene_head(h.float())
        ar = torch.arange(self.n_genes + 1, device=h.device)
        return lg.masked_fill(ar < prev_gene.unsqueeze(-1), float('-inf'))

    def variant_query(self, h, g):
        return self.var_q(torch.cat([h, self.gene_emb(g)], -1))

    def variant_logits(self, q, gene):
        """Logits of all variant tokens of ONE gene for queries q [m,d] -> [m, n_tok(gene)]."""
        s, e = self._gs[gene], self._ge[gene]
        with torch.autocast(device_type=q.device.type, enabled=False):
            O = self.var_out(self.Pbar_tok[s:e]) + self.var_res[s:e]
            return q.float() @ O.t() + self.var_bias[s:e]

    def vaf_logits(self, h, g, conc):
        with torch.autocast(device_type=h.device.type, enabled=False):
            return self.vaf_head(torch.cat([h.float(), self.gene_emb(g), self.P_conc[conc]], -1))

    # ------------------------------------------------------------------ training forward
    def forward(self, batch):
        """batch: dis [B]; gene, tok, conc, vbin [B,T] (valid for j < n, tok may be -1 = unmappable);
        n [B]; sig [B] required iff n_sig > 0. Returns dict of scalar losses (mean per step) and accuracy counts."""
        dis, gene, tok, conc, vbin, n = (batch[k] for k in ('dis', 'gene', 'tok', 'conc', 'vbin', 'n'))
        sig = batch['sig'] if self.n_sig > 0 else None                     # KeyError if a signature model lacks it
        B, T = gene.shape
        dev = gene.device
        j = torch.arange(T, device=dev)[None]
        alt_valid = j < n[:, None]                                         # [B,T]
        x_alt = self.alt_embed(conc.clamp_min(0), vbin.clamp_min(0))
        x = torch.cat([self.prefix_embed(dis, sig)[:, None], x_alt], 1)    # [B,T+1,d]
        jj = torch.arange(T + 1, device=dev)[None]
        key_pad = jj > n[:, None]                                           # inputs 0..n are real
        h = self.encode(x, key_pad)                                         # [B,T+1,d]

        # gene targets at positions 0..n: gene_j for j<n, END at j=n
        tgt_gene = torch.full((B, T + 1), -100, dtype=torch.long, device=dev)
        tgt_gene[:, :T] = torch.where(alt_valid, gene, tgt_gene[:, :T])
        tgt_gene.scatter_(1, n[:, None], self.END)
        capped = n >= self.max_len - 1              # 127 alterations: END is an artefact of run8's truncation
        if capped.any():
            tgt_gene[capped, n[capped]] = -100
        prev = torch.full((B, T + 1), -1, dtype=torch.long, device=dev)
        prev[:, 1:] = torch.where(alt_valid, gene, torch.full_like(gene, -1))
        glog = self.gene_logits(h, prev)
        sel = tgt_gene >= 0
        loss_gene = F.cross_entropy(glog[sel], tgt_gene[sel])
        acc_gene = (glog[sel].argmax(-1) == tgt_gene[sel]).float().sum()
        n_gene = sel.sum()

        # chained heads at alteration steps (h at input position j predicts alteration j)
        hb, hj = torch.nonzero(alt_valid, as_tuple=True)
        h_s = h[hb, hj]
        g_s, t_s, c_s, v_s = gene[hb, hj], tok[hb, hj], conc[hb, hj], vbin[hb, hj]
        vlog = self.vaf_logits(h_s, g_s, c_s)
        loss_vaf = F.cross_entropy(vlog, v_s)
        acc_vaf = (vlog.argmax(-1) == v_s).float().sum()
        n_vaf = torch.tensor(v_s.numel(), device=dev)

        m = t_s >= 0
        q = self.variant_query(h_s[m], g_s[m])
        g_m, t_m = g_s[m], t_s[m]
        order = torch.argsort(g_m)
        g_sorted = g_m[order]
        uniq, counts = torch.unique_consecutive(g_sorted, return_counts=True)
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
        return {'loss': loss_gene + loss_var + loss_vaf, 'loss_gene': loss_gene, 'loss_var': loss_var,
                'loss_vaf': loss_vaf, 'acc_gene': acc_gene, 'n_gene': n_gene, 'acc_var': acc_var,
                'n_var': n_var, 'acc_vaf': acc_vaf, 'n_vaf': n_vaf}

    # ------------------------------------------------------------------ sampling
    def _sample(self, logits, temperature, generator):
        if temperature <= 0:
            return logits.argmax(-1)
        p = torch.softmax(logits / temperature, -1)
        return torch.multinomial(p, 1, generator=generator)[:, 0]

    def _draw_pool(self, tok, generator):
        """Concrete variant for bucket tokens tok [m] ~ bucket training frequencies."""
        a, b = self.pool_ptr[tok], self.pool_ptr[tok + 1]
        cum = self.pool_cum
        lo = torch.where(a > 0, cum[(a - 1).clamp_min(0)], torch.zeros_like(cum[a]))
        hi = cum[b - 1]
        u = torch.rand(tok.shape[0], device=tok.device, dtype=torch.float64, generator=generator)
        target = lo + u * (hi - lo)
        pos = torch.searchsorted(cum, target, right=True)
        pos = torch.minimum(torch.maximum(pos, a), b - 1)
        return self.pool_conc[pos]

    # ---------------------------------------------------------------- repeat rule (sampling only)
    def _valid_tokens(self, used, g, norep):
        """Host-side, ONE patient re-picking gene g. `used` = concretes this patient already has in gene g.
        A concrete flagged in `norep` may not appear twice. Returns (valid mask over g's tokens,
        set of used non-repeatable concretes)."""
        s, e = self._gs[g], self._ge[g]
        valid = np.ones(e - s, dtype=bool)
        used_nr = {int(c) for c in used if norep[c]}
        if not used_nr:
            return valid, used_nr
        tk, ct, pp, pc = self._np['tok_kind'], self._np['conc_tok'], self._np['pool_ptr'], self._np['pool_conc']
        buckets = set()
        for c in used_nr:
            t_ = ct[c]
            if tk[t_] == 0:
                valid[t_ - s] = False                               # frequent token of a used specific variant
            else:
                buckets.add(int(t_))
        for b in buckets:                                            # bucket exhausted: every member used
            if all(int(m) in used_nr for m in pc[pp[b]:pp[b + 1]]):
                valid[b - s] = False
        return valid, used_nr

    @torch.no_grad()
    def generate(self, dis, temperature=1.0, max_alts=127, norep=None, generator=None, sig=None):
        """dis [B] disease level ids (or NO_DISEASE); sig [B] FMI signature ids (signature models only;
        None = NO_SIGNATURE). Returns dict of [B,max_alts] long tensors
        gene/tok/conc/vbin (-1 padded), n [B], truncated [B] bool (hit max_alts without END).

        norep: optional bool array [n_concrete]; a flagged concrete can appear at most once per patient.
        Because genes are generated in non-decreasing order, only the PREVIOUS gene can be picked again,
        so the rule only acts when a patient re-picks its previous gene: (1) that gene is blocked if it has
        no valid token left, (2) its used frequent tokens / exhausted buckets are masked, (3) bucket draws
        that hit a used flagged concrete are redrawn among the remaining members. norep=None = no rule
        (identical to the unconstrained sampler)."""
        assert not self.training, 'call model.eval() before generate()'
        if norep is not None:
            assert self._np['conc_tok'] is not None, 'repeat rule needs conc_tok (pass it to the constructor)'
            norep = np.asarray(norep.cpu() if torch.is_tensor(norep) else norep, dtype=bool)
        B = dis.shape[0]
        dev = dis.device
        X = torch.zeros(B, max_alts + 1, self.d, device=dev)
        X[:, 0] = self.prefix_embed(dis, sig)
        out = {k: torch.full((B, max_alts), -1, dtype=torch.long, device=dev) for k in ('gene', 'tok', 'conc', 'vbin')}
        n = torch.full((B,), max_alts, dtype=torch.long, device=dev)
        alive = torch.ones(B, dtype=torch.bool, device=dev)
        prev = torch.full((B,), -1, dtype=torch.long, device=dev)
        truncated = torch.zeros(B, dtype=torch.bool, device=dev)
        if norep is not None:                                        # host mirrors for the rule
            h_gene = np.full((B, max_alts), -1, np.int64)
            h_conc = np.full((B, max_alts), -1, np.int64)
            h_prev = np.full(B, -1, np.int64)
            gs_np, ge_np = np.asarray(self._gs), np.asarray(self._ge)
        for t in range(max_alts + 1):
            if t == max_alts:                                                     # cap reached: force stop
                truncated = alive.clone()
                break
            h = self.encode(X[:, :t + 1])[:, t]                                  # [B,d]
            glog = self.gene_logits(h, prev).masked_fill(~self.gene_ok_ext, float('-inf'))
            if norep is not None and t > 0:
                # (1) block re-picking the previous gene when none of its tokens is valid any more.
                # #invalid tokens <= #alterations already in that gene, so only check when ntok <= used.
                has = h_prev >= 0
                used_in_prev = (h_gene[:, :t] == h_prev[:, None]).sum(1)
                ntok = np.where(has, ge_np[np.maximum(h_prev, 0)] - gs_np[np.maximum(h_prev, 0)], 0)
                cand = np.nonzero(has & (ntok <= used_in_prev) & alive.cpu().numpy())[0]
                blocked = [r for r in cand
                           if not self._valid_tokens(h_conc[r, :t][h_gene[r, :t] == h_prev[r]], h_prev[r], norep)[0].any()]
                if blocked:
                    rb = torch.as_tensor(blocked, device=dev)
                    glog[rb, prev[rb]] = float('-inf')
            g = self._sample(glog, temperature, generator)
            stop = alive & (g == self.END)
            n[stop] = t
            alive = alive & ~stop
            if not alive.any():
                break
            rows = torch.nonzero(alive, as_tuple=True)[0]
            gr, hr = g[rows], h[rows]
            q = self.variant_query(hr, gr)
            tok = torch.empty_like(gr)
            order = torch.argsort(gr)
            uniq, counts = torch.unique_consecutive(gr[order], return_counts=True)
            if norep is not None and t > 0:
                rows_np, gr_np = rows.cpu().numpy(), gr.cpu().numpy()
                order_np = order.cpu().numpy()
            ptr = 0
            for gg, cnt in zip(uniq.tolist(), counts.tolist()):
                idx = order[ptr:ptr + cnt]
                lg = self.variant_logits(q[idx], gg)
                if norep is not None and t > 0:
                    # (2) mask used specific variants for patients re-picking their previous gene
                    for li, i in enumerate(order_np[ptr:ptr + cnt]):
                        r = rows_np[i]
                        if h_prev[r] != gg:
                            continue
                        valid, _ = self._valid_tokens(h_conc[r, :t][h_gene[r, :t] == gg], gg, norep)
                        if not valid.all():
                            lg[li, torch.as_tensor(~valid, device=dev)] = float('-inf')
                ptr += cnt
                tok[idx] = self._gs[gg] + self._sample(lg, temperature, generator)
            conc = torch.where(self.tok_kind[tok] == 0, self.tok_conc[tok], torch.zeros_like(tok))
            bk = self.tok_kind[tok] == 1
            if bk.any():
                conc[bk] = self._draw_pool(tok[bk], generator)
            if norep is not None and t > 0 and bk.any():
                # (3) a bucket draw must not duplicate a used specific variant (only possible on a re-pick)
                bk_np, conc_np, tok_np = bk.cpu().numpy(), conc.cpu().numpy(), tok.cpu().numpy()
                for i in np.nonzero(bk_np & (gr_np == h_prev[rows_np]))[0]:
                    r = rows_np[i]
                    _, used_nr = self._valid_tokens(h_conc[r, :t][h_gene[r, :t] == gr_np[i]], gr_np[i], norep)
                    if int(conc_np[i]) not in used_nr:
                        continue
                    b = int(tok_np[i])
                    pp = self._np['pool_ptr']
                    mem = self._np['pool_conc'][pp[b]:pp[b + 1]]
                    w = self._np['pool_w'][pp[b]:pp[b + 1]] * np.array([int(m) not in used_nr for m in mem])
                    cs = np.cumsum(w)
                    if not cs[-1] > 0:
                        raise RuntimeError(f'exhausted bucket {b} was sampled')      # (2) should make this impossible
                    u = float(torch.rand(1, device=dev, dtype=torch.float64, generator=generator)) * cs[-1]
                    conc[i] = int(mem[min(np.searchsorted(cs, u, side='right'), len(mem) - 1)])
            vb = self._sample(self.vaf_logits(hr, gr, conc), temperature, generator)
            out['gene'][rows, t], out['tok'][rows, t] = gr, tok
            out['conc'][rows, t], out['vbin'][rows, t] = conc, vb
            X[rows, t + 1] = self.alt_embed(conc, vb)
            prev[rows] = gr
            if norep is not None:
                rn = rows.cpu().numpy()
                h_gene[rn, t] = gr.cpu().numpy()
                h_conc[rn, t] = conc.cpu().numpy()
                h_prev[rn] = h_gene[rn, t]
        out['n'] = n
        out['truncated'] = truncated
        return out


def build_decoder(vocab, emb, init=None, dropout=0.1, n_sig=0):
    """Construct a PatientDecoder from prep_data.py outputs (numpy vocab dict, embeddings dict).
    n_sig > 0 adds FMI signature conditioning (prep_signature.py labels)."""
    tok_logprior = torch.log(torch.as_tensor(vocab['tok_occ'], dtype=torch.float64)).float()
    lc = (init or {}).get('layer_cfg', {'d_model': 512, 'nhead': 8, 'dim_feedforward': 512,
                                         'num_layers': 4, 'layer_norm_eps': 1e-5})
    model = PatientDecoder(
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
