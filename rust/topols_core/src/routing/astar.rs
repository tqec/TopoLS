//! Grid A* (three variants, as in `topols/routing/astar.py`).
//!
//! Every expansion (heap pop) adds one unit to the per-thread `WORK` counter,
//! the unit in which search budgets are expressed; every variant gives up
//! after a fixed number of expansions.

use std::cell::Cell as StdCell;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::geometry::{Cell, Floors};

pub type Occ = FxHashSet<Cell>;

thread_local! {
    /// Work done by this thread, in A* expansions (plus fixed costs added by
    /// the embedding). Mirrors `astar.WORK[0]`.
    pub static WORK: StdCell<u64> = const { StdCell::new(0) };
}

#[inline]
pub fn work() -> u64 {
    WORK.with(|w| w.get())
}
#[inline]
pub fn add_work(n: u64) {
    WORK.with(|w| w.set(w.get() + n));
}

/// Calibration shared with Python (`astar.WORK_PER_SECOND` etc.).
pub const WORK_PER_SECOND: u64 = 245_000;
pub const NEXT_STATE_COST: u64 = 3;
pub const ASTAR_MAX_EXPANSIONS: u32 = (0.1 * WORK_PER_SECOND as f64) as u32;
pub const ASTAR_BASE_MAX_EXPANSIONS: u32 = (1e-3 * WORK_PER_SECOND as f64) as u32;

const DIRECTIONS: [Cell; 6] = [
    Cell::new(1, 0, 0),
    Cell::new(-1, 0, 0),
    Cell::new(0, 1, 0),
    Cell::new(0, -1, 0),
    Cell::new(0, 0, 1),
    Cell::new(0, 0, -1),
];
const DIRECTIONS_2D: [Cell; 4] = [Cell::new(1, 0, 0), Cell::new(-1, 0, 0), Cell::new(0, 1, 0), Cell::new(0, -1, 0)];

/// Heap entry ordered like Python's `(f, g, node, parent)` tuple.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct Entry {
    f: i64,
    g: i64,
    cell: Cell,
    parent: Option<Cell>,
}

/// Common constraints of the 3D variants.
pub struct Constraints<'a> {
    pub occupied: &'a Occ,
    pub z_floor: f64,
    pub z_max: Option<f64>,
    pub floors: Floors,
    /// Positions of placed idles (the masked one already removed): the column
    /// above each is blocked.
    pub idle_cells: &'a [Cell],
    pub ceiling_z: Option<f64>,
    pub max_expansions: u32,
}

#[inline]
fn blocked_by_idle(q: Cell, idle_cells: &[Cell]) -> bool {
    idle_cells.iter().any(|c| q.x == c.x && q.y == c.y && q.z >= c.z)
}

fn reconstruct(back: &FxHashMap<Cell, Option<Cell>>, dst: Cell) -> Vec<Cell> {
    let mut path = vec![dst];
    let mut cur = dst;
    while let Some(Some(p)) = back.get(&cur) {
        path.push(*p);
        cur = *p;
    }
    path.reverse();
    path
}

/// A* from `src` to `dst` under `c`. Returns the cells from `src` to `dst`
/// inclusive, or None when no path is found within the expansion cap.
pub fn astar_3d(src: Cell, dst: Cell, c: &Constraints) -> Option<Vec<Cell>> {
    let mut open: BinaryHeap<Reverse<Entry>> = BinaryHeap::new();
    open.push(Reverse(Entry { f: src.manhattan(dst), g: 0, cell: src, parent: None }));
    let mut seen: FxHashMap<Cell, i64> = FxHashMap::default();
    seen.insert(src, 0);
    let mut back: FxHashMap<Cell, Option<Cell>> = FxHashMap::default();
    let mut count: u32 = 0;

    while let Some(Reverse(e)) = open.pop() {
        add_work(1);
        // Lazy deletion: skip entries superseded by a cheaper path.
        if e.g > seen[&e.cell] {
            continue;
        }
        back.insert(e.cell, e.parent);
        if e.cell == dst {
            return Some(reconstruct(&back, dst));
        }
        let p = e.cell;
        for d in DIRECTIONS {
            let q = p.add(d);
            if c.occupied.contains(&q)
                || ((q.z as f64) < c.z_floor && q != dst)
                || c.z_max.map_or(false, |zm| (q.z as f64) > zm)
                || (c.floors.outside(q) && q != dst)
            {
                continue;
            }
            if c.ceiling_z.map_or(false, |cz| (q.z as f64) > cz) {
                continue;
            }
            if blocked_by_idle(q, c.idle_cells) {
                continue;
            }
            let g2 = e.g + 1;
            if g2 < *seen.get(&q).unwrap_or(&i64::MAX) {
                seen.insert(q, g2);
                open.push(Reverse(Entry { f: g2 + q.manhattan(dst), g: g2, cell: q, parent: Some(p) }));
            }
        }
        count += 1;
        if count > c.max_expansions {
            return None;
        }
    }
    None
}

/// `shortest_path`: first bounded by the highest occupied z, then unbounded.
/// `occupied` must be non-empty (Python computes `max` over it).
pub fn shortest_path(
    src: Cell,
    dst: Cell,
    occupied: &Occ,
    z_floor: f64,
    floors: Floors,
    idle_cells: &[Cell],
    ceiling_z: Option<f64>,
    max_expansions: Option<u32>,
) -> Option<Vec<Cell>> {
    let max_expansions = max_expansions.unwrap_or(ASTAR_MAX_EXPANSIONS);
    let z_max = occupied.iter().map(|c| c.z).max().expect("shortest_path: empty occupancy") as f64;
    let mut c = Constraints { occupied, z_floor, z_max: Some(z_max), floors, idle_cells, ceiling_z, max_expansions };
    if let Some(p) = astar_3d(src, dst, &c) {
        return Some(p);
    }
    c.z_max = None;
    astar_3d(src, dst, &c)
}

/// `shortest_path_base`: A* within the plane `z = z_search`; `wall` holds
/// (x, y) columns that may not be crossed.
pub fn shortest_path_base(
    target_1: Cell,
    target_2: Cell,
    occupied: &Occ,
    wall: &FxHashSet<(i32, i32)>,
    z_search: i32,
    floors: Floors,
    max_expansions: Option<u32>,
) -> Option<Vec<Cell>> {
    let max_expansions = max_expansions.unwrap_or(ASTAR_BASE_MAX_EXPANSIONS);
    let t1 = Cell::new(target_1.x, target_1.y, z_search);
    let t2 = Cell::new(target_2.x, target_2.y, z_search);
    let mut open: BinaryHeap<Reverse<Entry>> = BinaryHeap::new();
    open.push(Reverse(Entry { f: t1.manhattan(t2), g: 0, cell: t1, parent: None }));
    let mut seen: FxHashMap<Cell, i64> = FxHashMap::default();
    seen.insert(t1, 0);
    let mut back: FxHashMap<Cell, Option<Cell>> = FxHashMap::default();
    let mut count: u32 = 0;

    while let Some(Reverse(e)) = open.pop() {
        add_work(1);
        if e.g > seen[&e.cell] {
            continue;
        }
        back.insert(e.cell, e.parent);
        if e.cell == t2 {
            return Some(reconstruct(&back, t2));
        }
        let p = e.cell;
        for d in DIRECTIONS_2D {
            let q = p.add(d);
            if occupied.contains(&q) || wall.contains(&(q.x, q.y)) || (floors.outside(q) && q != t2) {
                continue;
            }
            let g2 = e.g + 1;
            if g2 < *seen.get(&q).unwrap_or(&i64::MAX) {
                seen.insert(q, g2);
                open.push(Reverse(Entry { f: g2 + q.manhattan(t2), g: g2, cell: q, parent: Some(p) }));
            }
        }
        count += 1;
        if count > max_expansions {
            return None;
        }
    }
    None
}
