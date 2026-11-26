"""Helpers shared by ccf_prep.py and ccf_prep_armloss.py.
ccf_to_bin and purity: copied unchanged from run9 evolution_2/common2.py; gene_positions: copied unchanged from run9 evolution/genepos.py.
Only change: the input files are arguments instead of fixed paths."""
import gzip

import numpy as np
import pandas as pd

CCF_STEP, TOP_BIN = 0.05, 20                       # decoder bins; CCF >= 1 -> bin 20


def purity(sample_ids, samples_parquet):
    """FMI computationalTumorPurity (percent -> fraction) per sample id; NaN if missing.  samples_parquet: columns xrn, computationalTumorPurity."""
    s = pd.read_parquet(samples_parquet).set_index('xrn')
    return (pd.to_numeric(s.computationalTumorPurity, errors='coerce') / 100).reindex(pd.Index(sample_ids)).to_numpy()


def ccf_to_bin(ccf):
    c = np.minimum(np.asarray(ccf, float), 1.0)
    return np.minimum(np.floor(c / CCF_STEP + 1e-9).astype(int), TOP_BIN)


def gene_positions(refseq):
    """gene -> (chromosome, arm, start Mb) from RefSeq (UCSC ncbiRefSeq.txt.gz; approximate hg38 centromeres)."""
    CEN = {'1': 123.4, '2': 93.9, '3': 90.9, '4': 50.0, '5': 48.8, '6': 59.8, '7': 60.1, '8': 45.2, '9': 43.0, '10': 39.8,
           '11': 53.4, '12': 35.5, '13': 17.7, '14': 17.2, '15': 19.0, '16': 36.8, '17': 25.1, '18': 18.5, '19': 26.2,
           '20': 28.1, '21': 12.0, '22': 15.0, 'X': 60.6}
    pos = {}
    with gzip.open(refseq, 'rt') as fh:
        for ln in fh:
            f_ = ln.split('\t')
            c = f_[2][3:]
            if c in CEN:
                pos.setdefault(f_[12], []).append((c, int(f_[4]) / 1e6))
    out = {}
    for g, lst in pos.items():
        c = pd.Series([x[0] for x in lst]).mode()[0]
        st = float(np.median([x[1] for x in lst if x[0] == c]))
        out[g] = (c, 'p' if st < CEN[c] else 'q', st)
    return out
