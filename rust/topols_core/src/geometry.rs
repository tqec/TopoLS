//! Integer 3D cells and the small vector helpers the router needs.

/// A grid cell. Derived `Ord` is lexicographic in (x, y, z), matching the
/// comparison of Python tuples that the A* heap relies on for tie-breaking.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Default)]
pub struct Cell {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl Cell {
    pub const fn new(x: i32, y: i32, z: i32) -> Cell {
        Cell { x, y, z }
    }
    pub fn add(self, d: Cell) -> Cell {
        Cell::new(self.x + d.x, self.y + d.y, self.z + d.z)
    }
    pub fn neg(self) -> Cell {
        Cell::new(-self.x, -self.y, -self.z)
    }
    /// Displacement from `self` to `q` (Python `vector(p, q)`).
    pub fn vector_to(self, q: Cell) -> Cell {
        Cell::new(q.x - self.x, q.y - self.y, q.z - self.z)
    }
    pub fn manhattan(self, q: Cell) -> i64 {
        ((self.x - q.x).abs() + (self.y - q.y).abs() + (self.z - q.z).abs()) as i64
    }
    /// Cross product (for the corner normal in `color_switch`).
    pub fn cross(self, o: Cell) -> Cell {
        Cell::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }
}

/// Footprint limits (inclusive); `None` disables a limit. Python stores them
/// as floats (`k + 0.5`), so comparisons are done in f64.
#[derive(Clone, Copy, Debug, Default)]
pub struct Floors {
    pub x_min: Option<f64>,
    pub x_max: Option<f64>,
    pub y_min: Option<f64>,
    pub y_max: Option<f64>,
}

impl Floors {
    pub fn all(x_min: f64, x_max: f64, y_min: f64, y_max: f64) -> Floors {
        Floors { x_min: Some(x_min), x_max: Some(x_max), y_min: Some(y_min), y_max: Some(y_max) }
    }
    /// True iff `q` violates a limit (the caller exempts the destination).
    #[inline]
    pub fn outside(&self, q: Cell) -> bool {
        let (x, y) = (q.x as f64, q.y as f64);
        self.x_min.map_or(false, |m| x < m)
            || self.x_max.map_or(false, |m| x > m)
            || self.y_min.map_or(false, |m| y < m)
            || self.y_max.map_or(false, |m| y > m)
    }
    /// True iff `q` lies within every limit (all limits must be set; used by
    /// `color_switch`, whose floors are never None in Python).
    #[inline]
    pub fn inside_all(&self, q: Cell) -> bool {
        let (x, y) = (q.x as f64, q.y as f64);
        x >= self.x_min.unwrap() && x <= self.x_max.unwrap() && y >= self.y_min.unwrap() && y <= self.y_max.unwrap()
    }
}

/// The cells strictly between a path's two ends (Python `path[1:-1]`; empty
/// for paths shorter than two cells).
#[inline]
pub fn interior(path: &[Cell]) -> &[Cell] {
    if path.len() < 2 { &[] } else { &path[1..path.len() - 1] }
}
