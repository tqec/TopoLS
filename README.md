# TopoLS: Topological Lattice Surgery


TopoLS leverages the topological nature of lattice surgery to convert quantum circuit into corresponding logical circuit based on surface code.


## ✨ Overview

TopoLS performs compilation in three stages:

<p align="center">
  <img src="assets/overview.png" width="65%"/>
</p>

---

### 🟦 1. ZX-Level Topological Optimization

Quantum circuits are transformed into **ZX diagrams**, where
spider fusion simplifies the structure.

The resulting ZX graph is then **layer-sliced based on topological connectivity**, directly exposing merge–split operations and enabling
space–time volume reductions that are not visible in gate-based representations.

---

### 🟦 2. 3D Layout Optimization via MCTS

Using the enriched ZX diagram with layer information,
we apply **Monte Carlo Tree Search (MCTS)** to explore efficient
3D embeddings of operations.

MCTS guides compilation toward layouts with reduced
space–time volume by searching over embedding decisions.

---

### 🟦 3. Topology-Aware Circuit Partitioning


To ensure scalability, circuits are dynamically partitioned
based on spider connectivity.

This limits the number of operations per layer,
keeping the embedding problem tractable while preserving
topological optimization benefits.

---

Resources for TopoLS:

- 📄 **Paper**  
👉 [TopoLS: Lattice Surgery Compilation via Topological Program Transformations](https://arxiv.org/abs/2601.23109).

- 🎥 Video:  
👉 [TopoLS Presentation at TQEC](https://drive.google.com/file/d/12-Uby-_GgCEUzkFRkGJn-41uRcGZoh5H/view).

- 📊 Slide:  
👉 [TopoLS Slide](https://drive.google.com/file/d/1vOckwK4KiAtYmOgA3LbHbEPJ3BVxh2Ri/view?usp=sharing).

## 🚀 Examples

We demonstrate the compilation results of TopoLS on two representative cases: 
a 16-qubit GHZ state and a 500-qubit random circuit.

<p align="center">
  <img src="assets/example.png" width="75%"/>
</p>

A detailed usage tutorial is available in `src/tutorial.ipynb`.  
To reproduce the experimental results reported in the paper, run `src/exp.py`.

## 🔗 Operates with TQEC

TopoLS compiles circuits into a lattice-surgery pipe diagram, which can be directly consumed by TQEC for simulation and resource evaluation.

<p align="center">
  <img src="assets/simulation.png" width="80%"/>
</p>

## 🛠 Installation

We recommend installing TopoLS inside a virtual environment.

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  

# Clone the repository
python -m pip install git+https://github.com/tqec/TopoLS.git

# Install dependencies
pip install -r requirements.txt

```

## 📖 Citation

If you use **TopoLS** in your research, please cite this work:

```bibtex
@article{zhou2026topols,
  title={TopoLS: Lattice Surgery Compilation via Topological Program Transformations},
  author={Zhou, Junyu and Liu, Yuhao and Decker, Ethan and Kalloor, Justin and Weiden, Mathias and Chen, Kean and Iancu, Costin and Li, Gushu},
  journal={arXiv preprint arXiv:2601.23109},
  year={2026}
}