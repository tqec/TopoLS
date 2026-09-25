//! TopoLS embedding core.
//!
//! A faithful port of `topols.routing` / `topols.embedding` / `topols.driver`:
//! the layer-by-layer MCTS embedding of a layered ZX diagram into a 3D grid.
//! Every decision is deterministic and follows the Python implementation
//! exactly (same iteration orders, same RNG, same work-unit budget), so a
//! compile returns the same result as the Python reference.

pub mod geometry;
pub mod pyrandom;
pub mod routing;
