//! Colour algebra of pipes (`topols/routing/color_algebra.py`).
//!
//! A cube's colouring is an orientation (the axis whose faces carry the odd
//! colour) plus a type in {0, 1}; every bend of a pipe may change the type.

use rustc_hash::FxHashSet;

use crate::geometry::{Cell, Floors};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Axis {
    I,
    J,
    K,
}

impl Axis {
    pub fn of_step(d: Cell) -> Axis {
        match (d.x, d.y, d.z) {
            (1, 0, 0) | (-1, 0, 0) => Axis::I,
            (0, 1, 0) | (0, -1, 0) => Axis::J,
            (0, 0, 1) | (0, 0, -1) => Axis::K,
            _ => panic!("not a unit step: {:?}", d),
        }
    }
    pub fn as_char(self) -> char {
        match self {
            Axis::I => 'i',
            Axis::J => 'j',
            Axis::K => 'k',
        }
    }
    pub fn from_char(c: char) -> Axis {
        match c {
            'i' => Axis::I,
            'j' => Axis::J,
            'k' => Axis::K,
            _ => panic!("bad axis {c}"),
        }
    }
    /// Unit steps along the axis, both directions (`AXIS_OFFSETS`).
    pub fn offsets(self) -> [Cell; 2] {
        match self {
            Axis::I => [Cell::new(1, 0, 0), Cell::new(-1, 0, 0)],
            Axis::J => [Cell::new(0, 1, 0), Cell::new(0, -1, 0)],
            Axis::K => [Cell::new(0, 0, 1), Cell::new(0, 0, -1)],
        }
    }
}

/// `RULE_S[(face, ori, first_axis)]`: type after the first bend.
pub fn rule_s(face: u8, ori: Axis, first: Axis) -> u8 {
    use Axis::*;
    match (face, ori, first) {
        (0, I, J) | (0, I, K) | (0, J, I) => 0,
        (0, J, K) | (0, K, I) | (0, K, J) => 1,
        (1, I, J) | (1, I, K) | (1, J, I) => 1,
        (1, J, K) | (1, K, I) | (1, K, J) => 0,
        _ => panic!("RULE_S undefined for {:?}", (face, ori, first)),
    }
}

/// `RULES[(type, prev_axis, next_axis)]`: type after a subsequent bend.
pub fn rules(t: u8, frm: Axis, to: Axis) -> u8 {
    use Axis::*;
    match (t, frm, to) {
        (0, I, J) => 0,
        (0, I, K) => 1,
        (1, I, J) => 1,
        (1, I, K) => 0,
        (0, J, I) => 0,
        (0, J, K) => 0,
        (1, J, I) => 1,
        (1, J, K) => 1,
        (0, K, I) => 1,
        (0, K, J) => 0,
        (1, K, I) => 0,
        (1, K, J) => 1,
        _ => panic!("RULES undefined for {:?}", (t, frm, to)),
    }
}

/// `ORI_MAP[(arrival_axis, type, node_type)]`: orientation a cube must have
/// when a pipe arrives along `arrival` carrying colour `t`, for a cube of
/// `node_type` 0 (Z) or 1 (X).
pub fn ori_map(arrival: Axis, t: u8, node_type: u8) -> Axis {
    use Axis::*;
    match (arrival, t, node_type) {
        (I, 0, 0) => J,
        (I, 0, 1) => K,
        (I, 1, 0) => K,
        (I, 1, 1) => J,
        (J, 0, 0) => I,
        (J, 0, 1) => K,
        (J, 1, 0) => K,
        (J, 1, 1) => I,
        (K, 0, 0) => I,
        (K, 0, 1) => J,
        (K, 1, 0) => J,
        (K, 1, 1) => I,
        _ => panic!("ORI_MAP undefined for {:?}", (arrival, t, node_type)),
    }
}

/// Follow a pipe from a cube with colouring `(ori, face)` and report the
/// colour type carried into the last cell and the axis of the last segment.
pub fn edge_tracer(path: &[Cell], ori: Axis, face: u8) -> (u8, Axis) {
    assert!(path.len() >= 2, "Path must contain at least two points to form an edge.");
    // axes of the steps, consecutive equal axes collapsed
    let mut dirs: Vec<Axis> = Vec::with_capacity(path.len());
    for w in path.windows(2) {
        let a = Axis::of_step(w[0].vector_to(w[1]));
        if dirs.last() != Some(&a) {
            dirs.push(a);
        }
    }
    let mut t = rule_s(face, ori, dirs[0]);
    for w in dirs.windows(2) {
        t = rules(t, w[0], w[1]);
    }
    (t, *dirs.last().unwrap())
}

/// Change the colour a pipe delivers by inserting a two-cell detour next to
/// one of its corners; returns the new path or None.
pub fn color_switch(path: &[Cell], occupied: &FxHashSet<Cell>, z_floor: f64, floors: Floors) -> Option<Vec<Cell>> {
    let mut occ: FxHashSet<Cell> = occupied.clone();
    occ.extend(path.iter().copied());
    if path.len() < 5 {
        return None;
    }
    let legal = |c: Cell| !occ.contains(&c) && (c.z as f64) >= z_floor && floors.inside_all(c);
    for i in 2..path.len() - 2 {
        let (a, b, c) = (path[i - 1], path[i], path[i + 1]);
        let v_in = a.vector_to(b);
        let v_out = b.vector_to(c);
        if v_in == v_out {
            continue;
        }
        let face_dir = v_in.cross(v_out);

        // entry side of the corner
        let v_pre = path[i - 2].vector_to(a);
        let dirs: &[Cell] = if v_pre == v_in {
            &[face_dir, face_dir.neg()]
        } else if v_pre == face_dir {
            &[face_dir, v_out.neg()]
        } else if v_pre == face_dir.neg() {
            &[face_dir.neg(), v_out.neg()]
        } else {
            &[]
        };
        for &d in dirs {
            let (a_p, b_p) = (a.add(d), b.add(d));
            if legal(a_p) && legal(b_p) {
                let mut out = path[..i].to_vec();
                out.push(a_p);
                out.push(b_p);
                out.extend_from_slice(&path[i..]);
                return Some(out);
            }
        }

        // exit side of the corner
        let v_pos = c.vector_to(path[i + 2]);
        let dirs: &[Cell] = if v_pos == v_out {
            &[face_dir, face_dir.neg()]
        } else if v_pos == face_dir {
            &[face_dir.neg(), v_in]
        } else if v_pos == face_dir.neg() {
            &[face_dir, v_in]
        } else {
            &[]
        };
        for &d in dirs {
            let (b_p, c_p) = (b.add(d), c.add(d));
            if legal(b_p) && legal(c_p) {
                let mut out = path[..=i].to_vec();
                out.push(b_p);
                out.push(c_p);
                out.extend_from_slice(&path[i + 1..]);
                return Some(out);
            }
        }
    }
    None
}
