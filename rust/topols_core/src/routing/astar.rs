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

/// An occupancy the router can query. `Occ` itself, or an `OccView` that
/// applies a few removals and additions on top of a shared set without
/// copying it (Python builds `occ_tmp = set(occ) - {...} | {...}` per call).
pub trait Blocked {
    fn contains(&self, c: &Cell) -> bool;
    /// Highest z among the blocked cells (None if empty).
    fn max_z(&self) -> Option<i32>;
}

impl Blocked for Occ {
    #[inline]
    fn contains(&self, c: &Cell) -> bool {
        FxHashSet::contains(self, c)
    }
    fn max_z(&self) -> Option<i32> {
        self.iter().map(|c| c.z).max()
    }
}

/// `base - removed + extra` (removals first, so a cell in both counts as
/// present, like Python's remove-then-add).
pub struct OccView<'a> {
    pub base: &'a Occ,
    pub removed: &'a [Cell],
    pub extra: &'a [Cell],
}

impl Blocked for OccView<'_> {
    #[inline]
    fn contains(&self, c: &Cell) -> bool {
        self.extra.contains(c) || (self.base.contains(c) && !self.removed.contains(c))
    }
    fn max_z(&self) -> Option<i32> {
        let a = self.base.iter().filter(|c| !self.removed.contains(c)).map(|c| c.z).max();
        let b = self.extra.iter().map(|c| c.z).max();
        match (a, b) {
            (Some(x), Some(y)) => Some(x.max(y)),
            (x, None) => x,
            (None, y) => y,
        }
    }
}

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
pub struct Constraints<'a, B: Blocked> {
    pub occupied: &'a B,
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

/// Per-cell search record: best known g (`seen`) and, once the cell has
/// been expanded, the parent it was expanded from (`back`).
#[derive(Clone, Copy)]
struct Rec {
    g: i64,
    back: Option<Cell>,
    expanded: bool,
}

struct Buffers {
    open: BinaryHeap<Reverse<Entry>>,
    recs: FxHashMap<Cell, Rec>,
}

thread_local! {
    // Reused across calls: A* is called millions of times and the heap/map
    // allocations were a quarter of the run time.
    static BUFFERS: std::cell::RefCell<Buffers> = std::cell::RefCell::new(Buffers { open: BinaryHeap::new(), recs: FxHashMap::default() });
}

fn reconstruct(recs: &FxHashMap<Cell, Rec>, dst: Cell) -> Vec<Cell> {
    let mut path = vec![dst];
    let mut cur = dst;
    while let Some(p) = recs.get(&cur).and_then(|r| r.back) {
        path.push(p);
        cur = p;
    }
    path.reverse();
    path
}

/// A* from `src` to `dst` under `c`. Returns the cells from `src` to `dst`
/// inclusive, or None when no path is found within the expansion cap.
pub fn astar_3d<B: Blocked>(src: Cell, dst: Cell, c: &Constraints<B>) -> Option<Vec<Cell>> {
    BUFFERS.with(|b| {
        let mut b = b.borrow_mut();
        let Buffers { open, recs } = &mut *b;
        open.clear();
        recs.clear();
        open.push(Reverse(Entry { f: src.manhattan(dst), g: 0, cell: src, parent: None }));
        recs.insert(src, Rec { g: 0, back: None, expanded: false });
        let mut count: u32 = 0;

        while let Some(Reverse(e)) = open.pop() {
            add_work(1);
            // Lazy deletion: skip entries superseded by a cheaper path.
            let rec = recs.get_mut(&e.cell).unwrap();
            if e.g > rec.g {
                continue;
            }
            rec.back = e.parent;
            rec.expanded = true;
            if e.cell == dst {
                return Some(reconstruct(recs, dst));
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
                let better = match recs.get(&q) {
                    Some(r) => g2 < r.g,
                    None => true,
                };
                if better {
                    recs.entry(q).and_modify(|r| r.g = g2).or_insert(Rec { g: g2, back: None, expanded: false });
                    open.push(Reverse(Entry { f: g2 + q.manhattan(dst), g: g2, cell: q, parent: Some(p) }));
                }
            }
            count += 1;
            if count > c.max_expansions {
                return None;
            }
        }
        None
    })
}

/// `shortest_path`: first bounded by the highest occupied z, then unbounded.
/// `occupied` must be non-empty (Python computes `max` over it).
pub fn shortest_path<B: Blocked>(
    src: Cell,
    dst: Cell,
    occupied: &B,
    z_floor: f64,
    floors: Floors,
    idle_cells: &[Cell],
    ceiling_z: Option<f64>,
    max_expansions: Option<u32>,
) -> Option<Vec<Cell>> {
    let max_expansions = max_expansions.unwrap_or(ASTAR_MAX_EXPANSIONS);
    let z_max = occupied.max_z().expect("shortest_path: empty occupancy") as f64;
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
    BUFFERS.with(|b| {
        let mut b = b.borrow_mut();
        let Buffers { open, recs } = &mut *b;
        open.clear();
        recs.clear();
        open.push(Reverse(Entry { f: t1.manhattan(t2), g: 0, cell: t1, parent: None }));
        recs.insert(t1, Rec { g: 0, back: None, expanded: false });
        let mut count: u32 = 0;

        while let Some(Reverse(e)) = open.pop() {
            add_work(1);
            let rec = recs.get_mut(&e.cell).unwrap();
            if e.g > rec.g {
                continue;
            }
            rec.back = e.parent;
            rec.expanded = true;
            if e.cell == t2 {
                return Some(reconstruct(recs, t2));
            }
            let p = e.cell;
            for d in DIRECTIONS_2D {
                let q = p.add(d);
                if occupied.contains(&q) || wall.contains(&(q.x, q.y)) || (floors.outside(q) && q != t2) {
                    continue;
                }
                let g2 = e.g + 1;
                let better = match recs.get(&q) {
                    Some(r) => g2 < r.g,
                    None => true,
                };
                if better {
                    recs.entry(q).and_modify(|r| r.g = g2).or_insert(Rec { g: g2, back: None, expanded: false });
                    open.push(Reverse(Entry { f: g2 + q.manhattan(t2), g: g2, cell: q, parent: Some(p) }));
                }
            }
            count += 1;
            if count > max_expansions {
                return None;
            }
        }
        None
    })
}
