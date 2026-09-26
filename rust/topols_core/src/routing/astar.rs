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

pub use crate::routing::occ::Occ;

/// An occupancy the router can query. `Occ` itself, or an `OccView` that
/// applies a few removals and additions on top of a shared set without
/// copying it (Python builds `occ_tmp = set(occ) - {...} | {...}` per call).
pub trait Blocked {
    fn contains(&self, c: &Cell) -> bool;
    /// Highest z among the blocked cells (None if empty).
    fn max_z(&self) -> Option<i32>;
    /// Visit every blocked cell.
    fn for_each(&self, f: &mut dyn FnMut(Cell));
}

impl Blocked for Occ {
    #[inline]
    fn contains(&self, c: &Cell) -> bool {
        Occ::contains(self, c)
    }
    fn max_z(&self) -> Option<i32> {
        Occ::max_z(self)
    }
    fn for_each(&self, f: &mut dyn FnMut(Cell)) {
        for c in self.iter() {
            f(c);
        }
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
        let a = self.base.max_z_without(self.removed);
        let b = self.extra.iter().map(|c| c.z).max();
        match (a, b) {
            (Some(x), Some(y)) => Some(x.max(y)),
            (x, None) => x,
            (None, y) => y,
        }
    }
    fn for_each(&self, f: &mut dyn FnMut(Cell)) {
        for c in self.base.iter() {
            if !self.removed.contains(&c) {
                f(c);
            }
        }
        for c in self.extra {
            f(*c);
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

/// Diagnostics: A* calls and time spent in A* on this thread.
thread_local! {
    pub static ASTAR_CALLS: StdCell<u64> = const { StdCell::new(0) };
    pub static ASTAR_NANOS: StdCell<u64> = const { StdCell::new(0) };
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

/// A cell packed into 64 bits so that integer order equals the lexicographic
/// order of `(x, y, z)` (each coordinate offset to be positive, 21 bits).
#[inline]
fn pack(c: Cell) -> u64 {
    const OFF: i64 = 1 << 20;
    (((c.x as i64 + OFF) as u64) << 42) | (((c.y as i64 + OFF) as u64) << 21) | ((c.z as i64 + OFF) as u64)
}
#[inline]
fn unpack(p: u64) -> Cell {
    const OFF: i64 = 1 << 20;
    const M: u64 = (1 << 21) - 1;
    Cell::new(((p >> 42) as i64 - OFF) as i32, (((p >> 21) & M) as i64 - OFF) as i32, ((p & M) as i64 - OFF) as i32)
}
#[inline]
fn blocked_by_idle(q: Cell, idle_cells: &[Cell]) -> bool {
    idle_cells.iter().any(|c| q.x == c.x && q.y == c.y && q.z >= c.z)
}

/// Packed parent: 0 stands for Python's `None` (which sorts first).
const NO_PARENT: u64 = 0;

/// Heap entry ordered like Python's `(f, g, node, parent)` tuple.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct Entry {
    f: i32,
    g: i32,
    cell: u64,
    parent: u64,
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

/// Per-cell search record: best known g (`seen`) and, once the cell has
/// been expanded, the parent it was expanded from (`back`).
#[derive(Clone, Copy)]
struct Rec {
    gen: u32,
    g: i32,
    back: u64,
    has_back: bool,
}
const EMPTY: Rec = Rec { gen: 0, g: 0, back: NO_PARENT, has_back: false };

/// Visited-cell records on a dense grid over the search box (a generation
/// stamp makes clearing O(1)); cells outside the box go to a small map.
struct Grid {
    x0: i32,
    y0: i32,
    z0: i32,
    nx: usize,
    ny: usize,
    nz: usize,
    gen: u32,
    cells: Vec<Rec>,
    overflow: FxHashMap<u64, Rec>,
}

impl Grid {
    fn new() -> Grid {
        Grid { x0: 0, y0: 0, z0: 0, nx: 0, ny: 0, nz: 0, gen: 0, cells: Vec::new(), overflow: FxHashMap::default() }
    }
    /// Start a search over the box `[x0, x0+nx) x [y0, y0+ny) x [z0, z0+nz)`.
    fn reset(&mut self, x0: i32, y0: i32, z0: i32, nx: usize, ny: usize, nz: usize) {
        let n = nx * ny * nz;
        if self.cells.len() < n {
            self.cells.resize(n, EMPTY);
        }
        self.gen = self.gen.wrapping_add(1);
        if self.gen == 0 {
            self.cells.fill(EMPTY);
            self.gen = 1;
        }
        self.x0 = x0;
        self.y0 = y0;
        self.z0 = z0;
        self.nx = nx;
        self.ny = ny;
        self.nz = nz;
        self.overflow.clear();
    }
    #[inline]
    fn index(&self, c: Cell) -> Option<usize> {
        let (x, y, z) = (c.x - self.x0, c.y - self.y0, c.z - self.z0);
        if x < 0 || y < 0 || z < 0 || x as usize >= self.nx || y as usize >= self.ny || z as usize >= self.nz {
            None
        } else {
            Some(((x as usize) * self.ny + y as usize) * self.nz + z as usize)
        }
    }
    /// Slot of `c` in the dense box (None: outside, use the overflow map).
    #[inline]
    fn slot(&self, c: Cell) -> Option<usize> {
        self.index(c)
    }
    #[inline]
    fn get_slot(&self, i: usize) -> Option<Rec> {
        let r = self.cells[i];
        if r.gen == self.gen { Some(r) } else { None }
    }
    #[inline]
    fn set_slot(&mut self, i: usize, r: Rec) {
        self.cells[i] = Rec { gen: self.gen, ..r };
    }
    #[inline]
    fn get(&self, c: Cell) -> Option<Rec> {
        match self.index(c) {
            Some(i) => {
                let r = self.cells[i];
                if r.gen == self.gen { Some(r) } else { None }
            }
            None => self.overflow.get(&pack(c)).copied(),
        }
    }
    #[inline]
    fn set(&mut self, c: Cell, r: Rec) {
        let r = Rec { gen: self.gen, ..r };
        match self.index(c) {
            Some(i) => self.cells[i] = r,
            None => {
                self.overflow.insert(pack(c), r);
            }
        }
    }
}

struct Buffers {
    open: BinaryHeap<Reverse<Entry>>,
    grid: Grid,
    /// blocked columns: lowest idle z per (x, y), on the grid's x/y box
    idle_min_z: Vec<i32>,
    /// occupancy rasterised over the grid box, one bit per cell
    occ_bits: Vec<u64>,
}

thread_local! {
    // Reused across calls: A* is called millions of times.
    static BUFFERS: std::cell::RefCell<Buffers> = std::cell::RefCell::new(Buffers { open: BinaryHeap::new(), grid: Grid::new(), idle_min_z: Vec::new(), occ_bits: Vec::new() });
}

fn reconstruct(grid: &Grid, dst: Cell) -> Vec<Cell> {
    let mut path = vec![dst];
    let mut cur = dst;
    loop {
        match grid.get(cur) {
            Some(r) if r.has_back && r.back != NO_PARENT => {
                cur = unpack(r.back);
                path.push(cur);
            }
            _ => break,
        }
    }
    path.reverse();
    path
}

/// Integer form of the limits a neighbour must satisfy (floors are
/// half-integers, so `x < x_min` is `x <= floor(x_min)`).
struct Bounds {
    x_lo: i32,
    x_hi: i32,
    y_lo: i32,
    y_hi: i32,
    z_lo: i32,
    z_hi: i32,
}

fn bounds(c: &Constraints<impl Blocked>) -> Bounds {
    let lo = |f: Option<f64>| f.map_or(i32::MIN, |v| v.ceil() as i32);
    let hi = |f: Option<f64>| f.map_or(i32::MAX, |v| v.floor() as i32);
    let mut z_hi = i32::MAX;
    if let Some(zm) = c.z_max {
        z_hi = z_hi.min(zm.floor() as i32);
    }
    if let Some(cz) = c.ceiling_z {
        z_hi = z_hi.min(cz.floor() as i32);
    }
    Bounds { x_lo: lo(c.floors.x_min), x_hi: hi(c.floors.x_max), y_lo: lo(c.floors.y_min), y_hi: hi(c.floors.y_max), z_lo: c.z_floor.ceil() as i32, z_hi }
}

/// A* from `src` to `dst` under `c`. Returns the cells from `src` to `dst`
/// inclusive, or None when no path is found within the expansion cap.
pub fn astar_3d<B: Blocked>(src: Cell, dst: Cell, c: &Constraints<B>) -> Option<Vec<Cell>> {
    ASTAR_CALLS.with(|n| n.set(n.get() + 1));
    astar_3d_inner(src, dst, c)
}

fn astar_3d_inner<B: Blocked>(src: Cell, dst: Cell, c: &Constraints<B>) -> Option<Vec<Cell>> {
    let b = bounds(c);
    BUFFERS.with(|buf| {
        let mut buf = buf.borrow_mut();
        let Buffers { open, grid, idle_min_z, .. } = &mut *buf;
        open.clear();

        // search box: the footprint (plus a margin for out-of-footprint
        // destinations), z from the floor to the ceiling / highest occupied z
        let (xa, xb) = (b.x_lo.min(src.x).min(dst.x) - 1, b.x_hi.max(src.x).max(dst.x) + 1);
        let (ya, yb) = (b.y_lo.min(src.y).min(dst.y) - 1, b.y_hi.max(src.y).max(dst.y) + 1);
        let za = b.z_lo.min(src.z).min(dst.z) - 1;
        let zb = if b.z_hi == i32::MAX { src.z.max(dst.z).max(b.z_lo) + 64 } else { b.z_hi.max(src.z).max(dst.z) + 1 };
        let (nx, ny, nz) = ((xb - xa + 1) as usize, (yb - ya + 1) as usize, (zb - za + 1) as usize);
        grid.reset(xa, ya, za, nx, ny, nz);

        // blocked columns: lowest idle z per (x, y) of the box
        idle_min_z.clear();
        idle_min_z.resize(nx * ny, i32::MAX);
        for ic in c.idle_cells {
            let (x, y) = (ic.x - xa, ic.y - ya);
            if x >= 0 && y >= 0 && (x as usize) < nx && (y as usize) < ny {
                let i = x as usize * ny + y as usize;
                idle_min_z[i] = idle_min_z[i].min(ic.z);
            }
        }
        let column_blocked = |q: Cell| -> bool {
            let (x, y) = (q.x - xa, q.y - ya);
            if x >= 0 && y >= 0 && (x as usize) < nx && (y as usize) < ny {
                q.z >= idle_min_z[x as usize * ny + y as usize]
            } else {
                blocked_by_idle(q, c.idle_cells)
            }
        };

        open.push(Reverse(Entry { f: src.manhattan(dst) as i32, g: 0, cell: pack(src), parent: NO_PARENT }));
        grid.set(src, Rec { gen: 0, g: 0, back: NO_PARENT, has_back: false });
        let mut count: u32 = 0;

        while let Some(Reverse(e)) = open.pop() {
            add_work(1);
            let p = unpack(e.cell);
            let mut rec = grid.get(p).unwrap();
            // Lazy deletion: skip entries superseded by a cheaper path.
            if e.g > rec.g {
                continue;
            }
            rec.back = e.parent;
            rec.has_back = true;
            grid.set(p, rec);
            if p == dst {
                return Some(reconstruct(grid, dst));
            }
            for d in DIRECTIONS {
                let q = p.add(d);
                if q != dst && (q.z < b.z_lo || q.x < b.x_lo || q.x > b.x_hi || q.y < b.y_lo || q.y > b.y_hi) {
                    continue;
                }
                if q.z > b.z_hi || c.occupied.contains(&q) || column_blocked(q) {
                    continue;
                }
                let g2 = e.g + 1;
                match grid.slot(q) {
                    Some(i) => {
                        let prev = grid.get_slot(i);
                        if prev.map_or(true, |r| g2 < r.g) {
                            let prev = prev.unwrap_or(EMPTY);
                            grid.set_slot(i, Rec { gen: 0, g: g2, back: prev.back, has_back: prev.has_back });
                            open.push(Reverse(Entry { f: g2 + q.manhattan(dst) as i32, g: g2, cell: pack(q), parent: e.cell }));
                        }
                    }
                    None => {
                        let prev = grid.get(q);
                        if prev.map_or(true, |r| g2 < r.g) {
                            let prev = prev.unwrap_or(EMPTY);
                            grid.set(q, Rec { gen: 0, g: g2, back: prev.back, has_back: prev.has_back });
                            open.push(Reverse(Entry { f: g2 + q.manhattan(dst) as i32, g: g2, cell: pack(q), parent: e.cell }));
                        }
                    }
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
    let lo = |f: Option<f64>| f.map_or(i32::MIN, |v| v.ceil() as i32);
    let hi = |f: Option<f64>| f.map_or(i32::MAX, |v| v.floor() as i32);
    let (x_lo, x_hi, y_lo, y_hi) = (lo(floors.x_min), hi(floors.x_max), lo(floors.y_min), hi(floors.y_max));
    BUFFERS.with(|buf| {
        let mut buf = buf.borrow_mut();
        let Buffers { open, grid, .. } = &mut *buf;
        open.clear();
        let (xa, xb) = (x_lo.min(t1.x).min(t2.x) - 1, x_hi.max(t1.x).max(t2.x) + 1);
        let (ya, yb) = (y_lo.min(t1.y).min(t2.y) - 1, y_hi.max(t1.y).max(t2.y) + 1);
        grid.reset(xa, ya, z_search, (xb - xa + 1) as usize, (yb - ya + 1) as usize, 1);
        open.push(Reverse(Entry { f: t1.manhattan(t2) as i32, g: 0, cell: pack(t1), parent: NO_PARENT }));
        grid.set(t1, Rec { gen: 0, g: 0, back: NO_PARENT, has_back: false });
        let mut count: u32 = 0;

        while let Some(Reverse(e)) = open.pop() {
            add_work(1);
            let p = unpack(e.cell);
            let mut rec = grid.get(p).unwrap();
            if e.g > rec.g {
                continue;
            }
            rec.back = e.parent;
            rec.has_back = true;
            grid.set(p, rec);
            if p == t2 {
                return Some(reconstruct(grid, t2));
            }
            for d in DIRECTIONS_2D {
                let q = p.add(d);
                if occupied.contains(&q) || wall.contains(&(q.x, q.y)) {
                    continue;
                }
                if q != t2 && (q.x < x_lo || q.x > x_hi || q.y < y_lo || q.y > y_hi) {
                    continue;
                }
                let g2 = e.g + 1;
                let prev = grid.get(q);
                if prev.map_or(true, |r| g2 < r.g) {
                    let prev = prev.unwrap_or(EMPTY);
                    grid.set(q, Rec { gen: 0, g: g2, back: prev.back, has_back: prev.has_back });
                    open.push(Reverse(Entry { f: g2 + q.manhattan(t2) as i32, g: g2, cell: pack(q), parent: e.cell }));
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
