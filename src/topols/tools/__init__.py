"""Command-line tools that work on a compiled result (`result/topols/<name>.pkl`).

Run them from the directory that holds `benchmark/` and `result/` (the
`docs/` directory in this repository):

    python -m topols.tools.hadamard_check -f ghz_16 -b 20
    python -m topols.tools.viz_region -f qaoa_16 --zmin 40 --zmax 50 -o crop
    python -m topols.tools.pipe_sim -f CNOT
"""
