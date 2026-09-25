//! Routing: grid A*, the colour algebra of pipes, and the special routes
//! (ceiling lifts, T-gate exits).

pub mod astar;
pub mod boundary;
pub mod color;

pub use astar::{Occ, WORK};
