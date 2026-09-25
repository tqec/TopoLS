//! Node identities and types.
//!
//! Python uses ints for vertices of the main graph and strings for the rest:
//! `"<v>_<block>"` for a fallback block's vertices, `"<id>_old"` for a node
//! renamed by `ceiling()`. `NodeId` carries the same information and can
//! reproduce the Python spelling (needed for ordering and for export).

use std::fmt;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct NodeId {
    pub base: u32,
    /// Fallback block suffix, if any.
    pub block: Option<u16>,
    /// Renamed `_old` by a ceiling lift.
    pub old: bool,
}

impl NodeId {
    pub const fn int(v: u32) -> NodeId {
        NodeId { base: v, block: None, old: false }
    }
    pub const fn in_block(v: u32, block: u16) -> NodeId {
        NodeId { base: v, block: Some(block), old: false }
    }
    pub fn as_old(self) -> NodeId {
        NodeId { old: true, ..self }
    }
    /// `_strip` in hadamard.py: the id without its `_old`.
    pub fn stripped(self) -> NodeId {
        NodeId { old: false, ..self }
    }
    pub fn is_int(self) -> bool {
        self.block.is_none() && !self.old
    }
    /// The Python spelling.
    pub fn python_str(&self) -> String {
        let mut s = self.base.to_string();
        if let Some(b) = self.block {
            s.push('_');
            s.push_str(&b.to_string());
        }
        if self.old {
            s.push_str("_old");
        }
        s
    }
    /// `layering._node_key`: ints first, numerically; then strings, by their
    /// Python spelling.
    pub fn sort_key(&self) -> (u8, u32, String) {
        if self.is_int() {
            (0, self.base, String::new())
        } else {
            (1, 0, self.python_str())
        }
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.python_str())
    }
}

/// `layering.ordered_edges`.
pub fn ordered_edges(edges: impl IntoIterator<Item = (NodeId, NodeId)>) -> Vec<(NodeId, NodeId)> {
    let mut v: Vec<(NodeId, NodeId)> = edges.into_iter().collect();
    v.sort_by_cached_key(|(a, b)| (a.sort_key(), b.sort_key()));
    v
}

/// Node types: 0 Z spider, 1 X spider, 2 idle, 3 Hadamard box, 4 S, 5 T.
pub type NodeType = i8;

#[inline]
pub fn is_cube(t: NodeType) -> bool {
    matches!(t, 0 | 1 | 4 | 5)
}
#[inline]
pub fn is_chain(t: NodeType) -> bool {
    matches!(t, 2 | 3)
}
/// Colour type used when tracing from a cube: its own type for Z/X, 0 for S/T.
#[inline]
pub fn trace_type(t: NodeType) -> u8 {
    if matches!(t, 4 | 5) { 0 } else { t as u8 }
}
