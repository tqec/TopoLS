# TopoLS architecture

This document describes how the compiler is organised and the ideas each
part relies on. It is written for people who want to read or extend the
code; usage is covered in the top-level `README.md`.

## Pipeline

```
QASM circuit
   │  pyzx: Circuit.load / to_graph
   ▼
ZX diagram ──► zx_transform ──► layered ZX diagram ──► embedding ──► 3D pipe diagram ──► export
              (simplify,          (one layer = one         (MCTS places each        (TQEC .bgraph,
               partition,          time step; every         layer's spiders in       2D/3D renderings)
               layering)           qubit has a node          a 3D grid and routes
                                   in every layer)           the wires between them)
```

1. **ZX transformation** (`topols.zx_transform`). The circuit is turned
   into a ZX diagram, simplified (spider fusion, removal of degree-2 phase-0
   spiders), and cut into *blocks* of bounded size by topological
   connectivity. Within each block the diagram is simplified again, then
   *layered*: every spider gets a layer index so that neighbouring spiders
   sit in consecutive layers, and idle spiders are inserted on any wire that
   would otherwise skip a layer. Hadamard gates are dissolved at this stage
   (see *Hadamards* below).

2. **Embedding** (`topols.embedding`). Layers are embedded one at a time
   into a 3D grid where `z` is time and each spider becomes a cube. For a
   layer, a Monte Carlo Tree Search chooses the `(x, y, z)` of every spider;
   each placement is immediately followed by routing every wire to the
   previous layer with A\*, and a placement whose wires cannot be routed
   consistently is rejected. Independent search trials with different
   random seeds run in parallel and the lowest-volume result is kept.

3. **Export** (`topols.export`). The finished embedding (positions,
   orientations, types and routed paths) is converted into a TQEC
   `BlockGraph`, saved as `.bgraph`, and rendered either as a static image
   or an interactive HTML scene.

## Module map

| module | role |
|---|---|
| `pipeline.py` | `prepare_graph`: QASM file → simplified, layered ZX diagram (`PreparedGraph`), the front end used by `docs/prog.py` |
| `zx_transform/simplify.py` | ZX simplification: spider fusion (`zx_optimization`), `hadamard_box` / `dissolve_hadamard_boxes`, `delete_singular_nodes` |
| `zx_transform/partition.py` | topology-aware slicing of the circuit into blocks (`find_block`, `circuit_slicing`) |
| `zx_transform/layering.py` | layer indices (`layer_labeling`), idle insertion, per-layer connectivity (`layer_info`), output-port alignment |
| `embedding/state.py` | `EmbeddingState`: the immutable search state; `next_state` places one spider and routes its wires; `reward` seals a finished layer |
| `embedding/mcts.py` | the anytime MCTS over `EmbeddingState`s (UCT selection, greedy rollouts, best-so-far result) |
| `embedding/hadamard.py` | `HTable`: decides where a Hadamard flips the colour of a wire (see below) |
| `embedding/ports.py` | input ports of a block (`auto_ports`), lifting a finished layer to a common ceiling (`ceiling`), space-time volume |
| `embedding/fallback.py` | `basic_embedding`: the deterministic brute-force layout used when search fails |
| `driver.py` | `operation()`: the layer loop, seed parallelisation, the fallback ladder and block hand-off |
| `routing/astar.py` | grid A\* with occupancy, floor/ceiling and boundary constraints |
| `routing/color_algebra.py` | how a cube's colour orientation evolves along a routed path (`edge_tracer`, `ORI_MAP`) and how to fix a mismatch (`color_switch`) |
| `routing/boundary.py` | routing to the ceiling and to the boundary (T-gate exits) |
| `export/bgraph.py` | pipe diagram → TQEC `BlockGraph` (`build_pipe_diagram`, `save_bigraph`) |
| `tools/viz_region.py`, `tools/pipe_sim.py` | command-line tools on a compiled result: cropped interactive rendering, TQEC/sinter simulation (`python -m topols.tools.<name>`) |
| `export/visualize.py`, `export/visualize_interactive.py` | matplotlib and Plotly renderers with identical colour conventions |
| `geometry.py` | small vector helpers |

## Key ideas

### Colour consistency

A surface-code cube has two colours on its faces (X and Z boundaries),
encoded here as an orientation `ori ∈ {i, j, k}` (the axis carrying the
odd colour) together with a spider type. When a wire is routed from cube A
to cube B, `routing.color_algebra.edge_tracer` follows the path and
computes which orientation B must have for its faces to match; if B already
has a different orientation, `color_switch` tries to re-route the wire
through an extra bend that flips the colour. Every routing decision in
`EmbeddingState.next_state` goes through this check, so an embedding that
is accepted is colour-consistent by construction.

### Hadamards as a property of the wire

A Hadamard swaps the two colours of a wire, so in a pipe diagram it is
not a cube but a colour change somewhere along a pipe. TopoLS therefore
removes Hadamard boxes from the ZX diagram before layering and records
each one by `(qubit, row)` in an `HTable`. Whenever routing connects two
real (non-idle) spiders A and B, `HTable.needs_flip(A, B)` answers whether
an odd number of Hadamards lies on that stretch of wire and, if so, the
target colour is flipped before the consistency check. Because the answer
depends only on circuit coordinates, it is the same in every part of the
compiler (main search, fallback, re-layered blocks), which is what makes
it robust. A Hadamard that sits directly before an output port keeps its
own cube (`rematerialize_stranded_hadamards`) since there is no later
spider to carry the flip.

### Anytime search and search budget

`mcts()` runs until either `-i` iterations or the `-t` budget per call is
spent and returns the best complete layer embedding seen. The budget is
expressed in *work*: A* expansions (`routing.astar.WORK`) plus a fixed
cost per placement, calibrated so that `-t 1` is about one second on the
reference machine. Because neither the budget nor A*'s give-up rule uses
the wall clock, a compile is reproducible on any machine, and a faster
implementation of the same search finishes sooner rather than searching
more. Given the random seed and the starting state, the sequence of
iterations is deterministic, so a longer budget extends the same search
and cannot return a worse layer. Several seeds (`-s`) are searched in parallel processes and the
best is kept; each seed gets its own snapshot of the shuffled input order.

Selecting the best layer greedily can occasionally pick a state from which
the *next* layer has no valid embedding. `--backtrack k` keeps the other
seeds' results for the previous layer and, when the current layer fails,
retries it from up to `k` of those alternatives (plain search first, then
search from a lifted ceiling) before falling back further.

### The fallback ladder

If no seed finds an embedding for a layer, `driver.operation` escalates:

1. **ceiling retry** — lift every open wire of the previous layer to a
   fresh ceiling plane (`ports.ceiling`) and search again from there;
2. **gate-by-gate** — re-layer the current block one gate per layer and
   embed those thinner layers with the same search;
3. **brute force** — `fallback.basic_embedding` stacks the layer
   deterministically, one spider per time step.

Every compile ends with a *final seal*: the last layer's open wires are
lifted to one ceiling, the output ends are coloured (applying any trailing
Hadamard) and the ports are aligned, whichever rung produced the last
layer.

### Blocks

Long circuits are compiled block by block. At a block boundary the current
frontier is lifted to a ceiling and becomes the input ports of the next
block; a block that fell through to gate-by-gate hands over its final
frontier the same way. Output ports of all qubits are kept on one last
layer so that the final seal sees every wire.
