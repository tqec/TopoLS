"""Reproduce the benchmark table with the Rust core.

    python3 exp_rust.py                 # all three configurations
    python3 exp_rust.py full            # any subset of: full part place

Same benchmarks, configurations and search settings as exp.py, run with
`prog.py --engine rust` (the extension must be installed, see README:
`maturin develop --release -m rust/topols_core/Cargo.toml`). Results go to
`result/topols/result_<config>_rust.csv` and `result/topols/summary_rust.csv`.

Both engines produce the same embedding for the same input, so the volumes
below are the volumes of the Python implementation as well. Reference
numbers (Full-Opt, one machine, 16 cores; Rust runs include the pyzx front
end, which is still Python):

    benchmark   paper Full-Opt        this repo, Python      this repo, Rust
                volume   time         volume   time          volume   time
    bv_16          486    26.3 s        486    18 s            486    0.5 s
    dj_16          891    55.8 s        567    30 s            567    1.0 s
    grover_6     23240  2412.0 s      22015   585 s          22015     21 s
    qft_16       36531  3097.0 s      35154  1228 s          35154     56 s
    qpe_16       39447  3316.6 s      37989  1290 s          37989     63 s
    vqe_16        4212   510.8 s       3645   265 s           3645     14 s
    ghz_16         243    12.7 s        243    13 s            243    0.6 s
    wstate_16     8505   686.2 s       8100   170 s           8100      8 s
    qaoa_16       4374   402.6 s       4050   230 s           4050     12 s

Volumes: seven of nine below the paper's, two equal. Time: the Rust core is
15-45x faster than this repo's Python and 20-110x faster than the paper's
runs (the whole Full-Opt set takes about 3 minutes); for grover/qft/qpe a quarter of the remaining time is the Python
front end (circuit parsing and block partitioning).
"""

import sys

from exp import CONFIGS, main

if __name__ == "__main__":
    configs = [a for a in sys.argv[1:] if a in CONFIGS]
    main(configs or list(CONFIGS), engine="rust", csv_suffix="_rust")
