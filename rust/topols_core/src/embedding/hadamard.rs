//! `HTable`: where Hadamards flip the colour of a wire (hadamard.py).

use rustc_hash::{FxHashMap, FxHashSet};

use crate::embedding::node::NodeId;

#[derive(Clone, Debug, Default)]
pub struct HTable {
    /// qubit -> sorted rows of the Hadamards on that wire
    pub rows_by_qubit: FxHashMap<i64, Vec<f64>>,
    /// Hadamards between two qubits, as the unordered pair {(q, r), (q, r)}
    /// stored with the smaller (q, r) first
    pub cross: FxHashSet<((i64, f64bits), (i64, f64bits))>,
    /// node -> (qubit, row)
    pub qrow: FxHashMap<NodeId, (i64, f64)>,
}

/// f64 as bits so it can be hashed (rows are compared exactly, as in Python).
#[allow(non_camel_case_types)]
pub type f64bits = u64;

fn key(q: i64, r: f64) -> (i64, f64bits) {
    (q, r.to_bits())
}

/// `bisect.bisect_right`
fn bisect_right(rows: &[f64], x: f64) -> usize {
    rows.partition_point(|&r| r <= x)
}

impl HTable {
    pub fn add_cross(&mut self, a: (i64, f64), b: (i64, f64)) {
        let (ka, kb) = (key(a.0, a.1), key(b.0, b.1));
        // frozenset semantics: order-free membership
        let pair = if (a.0, a.1) <= (b.0, b.1) { (ka, kb) } else { (kb, ka) };
        self.cross.insert(pair);
    }

    fn cross_contains(&self, a: (i64, f64), b: (i64, f64)) -> bool {
        let (ka, kb) = (key(a.0, a.1), key(b.0, b.1));
        self.cross.contains(&(ka, kb)) || self.cross.contains(&(kb, ka))
    }

    /// True iff the wire from real node `a` to real node `b` carries an odd
    /// number of Hadamards. Unknown ids give false.
    pub fn needs_flip(&self, a: NodeId, b: NodeId) -> bool {
        let (Some(&qa), Some(&qb)) = (self.qrow.get(&a.stripped()), self.qrow.get(&b.stripped())) else {
            return false;
        };
        if qa.0 != qb.0 {
            return self.cross_contains(qa, qb);
        }
        let Some(rows) = self.rows_by_qubit.get(&qa.0) else { return false };
        if rows.is_empty() {
            return false;
        }
        let (lo, hi) = if qa.1 <= qb.1 { (qa.1, qb.1) } else { (qb.1, qa.1) };
        let n = bisect_right(rows, hi) - bisect_right(rows, lo);
        n % 2 == 1
    }

    /// True iff an odd number of Hadamards lies on `a`'s qubit strictly after `a`.
    pub fn needs_flip_to_end(&self, a: NodeId) -> bool {
        let Some(&qa) = self.qrow.get(&a.stripped()) else { return false };
        let Some(rows) = self.rows_by_qubit.get(&qa.0) else { return false };
        if rows.is_empty() {
            return false;
        }
        (rows.len() - bisect_right(rows, qa.1)) % 2 == 1
    }
}
