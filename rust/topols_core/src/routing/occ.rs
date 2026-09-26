//! `Occ`: the set of occupied cells as a dense bitmap over a bounding box.
//!
//! Python keeps a `set` of cells. The router tests membership millions of
//! times on a set that is copied for every placement, so a bitmap (a few
//! dozen words for one layer's footprint) is both faster to query and
//! cheaper to clone. The API mirrors the handful of set operations the
//! compiler uses; iteration order is never relied upon.

use crate::geometry::Cell;

#[derive(Clone, Default, Debug)]
pub struct Occ {
    x0: i32,
    y0: i32,
    z0: i32,
    nx: i32,
    ny: i32,
    nz: i32,
    bits: Vec<u64>,
    len: usize,
    /// highest occupied z and the number of cells at that z (0 when empty)
    zmax: i32,
    n_at_zmax: usize,
}

impl Occ {
    pub fn new() -> Occ {
        Occ::default()
    }

    #[inline]
    fn index(&self, c: &Cell) -> Option<usize> {
        let (x, y, z) = (c.x - self.x0, c.y - self.y0, c.z - self.z0);
        if x < 0 || y < 0 || z < 0 || x >= self.nx || y >= self.ny || z >= self.nz {
            None
        } else {
            Some(((x * self.ny + y) * self.nz + z) as usize)
        }
    }

    #[inline]
    pub fn contains(&self, c: &Cell) -> bool {
        match self.index(c) {
            Some(i) => self.bits[i >> 6] & (1u64 << (i & 63)) != 0,
            None => false,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Rebuild over a box that also holds `c` (with a margin so that growth
    /// is rare).
    fn grow_to(&mut self, c: &Cell) {
        let cells: Vec<Cell> = self.iter().collect();
        let (mut xa, mut xb, mut ya, mut yb, mut za, mut zb) = if self.nx == 0 {
            (c.x, c.x, c.y, c.y, c.z, c.z)
        } else {
            (self.x0, self.x0 + self.nx - 1, self.y0, self.y0 + self.ny - 1, self.z0, self.z0 + self.nz - 1)
        };
        if c.x < xa { xa = c.x - 2 }
        if c.x > xb { xb = c.x + 2 }
        if c.y < ya { ya = c.y - 2 }
        if c.y > yb { yb = c.y + 2 }
        if c.z < za { za = c.z - 1 }
        if c.z > zb { zb = c.z + 8 }
        if self.nx == 0 {
            xa -= 2; xb += 2; ya -= 2; yb += 2; za -= 1; zb += 8;
        }
        *self = Occ { x0: xa, y0: ya, z0: za, nx: xb - xa + 1, ny: yb - ya + 1, nz: zb - za + 1, bits: Vec::new(), len: 0, zmax: i32::MIN, n_at_zmax: 0 };
        let n = (self.nx * self.ny * self.nz) as usize;
        self.bits = vec![0u64; (n + 63) / 64];
        for cell in cells {
            self.insert(cell);
        }
    }

    /// Insert; true if the cell was not present.
    pub fn insert(&mut self, c: Cell) -> bool {
        let i = match self.index(&c) {
            Some(i) => i,
            None => {
                self.grow_to(&c);
                self.index(&c).unwrap()
            }
        };
        let (w, m) = (i >> 6, 1u64 << (i & 63));
        if self.bits[w] & m != 0 {
            false
        } else {
            self.bits[w] |= m;
            self.len += 1;
            if c.z > self.zmax {
                self.zmax = c.z;
                self.n_at_zmax = 1;
            } else if c.z == self.zmax {
                self.n_at_zmax += 1;
            }
            true
        }
    }

    /// Remove; true if the cell was present.
    pub fn remove(&mut self, c: &Cell) -> bool {
        match self.index(c) {
            Some(i) => {
                let (w, m) = (i >> 6, 1u64 << (i & 63));
                if self.bits[w] & m != 0 {
                    self.bits[w] &= !m;
                    self.len -= 1;
                    if c.z == self.zmax {
                        self.n_at_zmax -= 1;
                        if self.n_at_zmax == 0 {
                            // recompute (rare): the highest layer was emptied
                            self.zmax = self.iter().map(|c| c.z).max().unwrap_or(i32::MIN);
                            self.n_at_zmax = if self.zmax == i32::MIN { 0 } else { self.iter().filter(|c| c.z == self.zmax).count() };
                        }
                    }
                    true
                } else {
                    false
                }
            }
            None => false,
        }
    }

    /// The occupied cells (in an unspecified order).
    pub fn iter(&self) -> impl Iterator<Item = Cell> + '_ {
        let (ny, nz) = (self.ny as usize, self.nz as usize);
        self.bits.iter().enumerate().flat_map(move |(w, &word)| {
            let mut word = word;
            std::iter::from_fn(move || {
                if word == 0 {
                    return None;
                }
                let b = word.trailing_zeros() as usize;
                word &= word - 1;
                let i = w * 64 + b;
                let z = i % nz;
                let y = (i / nz) % ny;
                let x = i / (nz * ny);
                Some(Cell::new(self.x0 + x as i32, self.y0 + y as i32, self.z0 + z as i32))
            })
        })
    }

    /// Highest z of any occupied cell.
    #[inline]
    pub fn max_z(&self) -> Option<i32> {
        if self.len == 0 { None } else { Some(self.zmax) }
    }

    /// Highest z once the cells `removed` (which may or may not be present)
    /// are taken away.
    pub fn max_z_without(&self, removed: &[Cell]) -> Option<i32> {
        if self.len == 0 {
            return None;
        }
        let gone = removed.iter().filter(|c| c.z == self.zmax && self.contains(c)).count();
        // distinct cells only matter if duplicates are passed; callers pass distinct cells
        if gone < self.n_at_zmax {
            Some(self.zmax)
        } else {
            self.iter().filter(|c| !removed.contains(c)).map(|c| c.z).max()
        }
    }
}

impl Extend<Cell> for Occ {
    fn extend<I: IntoIterator<Item = Cell>>(&mut self, iter: I) {
        for c in iter {
            self.insert(c);
        }
    }
}

impl FromIterator<Cell> for Occ {
    fn from_iter<I: IntoIterator<Item = Cell>>(iter: I) -> Occ {
        let cells: Vec<Cell> = iter.into_iter().collect();
        let mut o = Occ::new();
        if let (Some(xa), Some(xb), Some(ya), Some(yb), Some(za), Some(zb)) = (
            cells.iter().map(|c| c.x).min(), cells.iter().map(|c| c.x).max(),
            cells.iter().map(|c| c.y).min(), cells.iter().map(|c| c.y).max(),
            cells.iter().map(|c| c.z).min(), cells.iter().map(|c| c.z).max(),
        ) {
            o = Occ { x0: xa - 2, y0: ya - 2, z0: za - 1, nx: xb - xa + 5, ny: yb - ya + 5, nz: zb - za + 10, bits: Vec::new(), len: 0, zmax: i32::MIN, n_at_zmax: 0 };
            let n = (o.nx * o.ny * o.nz) as usize;
            o.bits = vec![0u64; (n + 63) / 64];
        }
        for c in cells {
            o.insert(c);
        }
        o
    }
}

impl PartialEq for Occ {
    fn eq(&self, other: &Occ) -> bool {
        self.len == other.len && self.iter().all(|c| other.contains(&c))
    }
}
