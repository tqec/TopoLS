# TopoLS: Topological Lattice Surgery

<p align="center">
  TopoLS leverages the topological nature of lattice surgery to convert quantum circuit into corresponding logical circuit based on surface code.
</p>


## ✨ Overview

TopoLS performs compilation in three stages:

<p align="center">
  <img src="assets/overview.png" width="750"/>
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

All the details can be found in our paper  
👉 [TopoLS: Lattice Surgery Compilation via Topological Program Transformations](https://arxiv.org/abs/2601.23109).



