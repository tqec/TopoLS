//! Special routes (`topols/routing/boundary.py`): lifting a path off a
//! corner, vertical segments, ceiling lifts and T-gate exits.

use crate::geometry::{Cell, Floors};
use crate::routing::astar::{shortest_path, Occ};
use crate::routing::color::Axis;

/// Raise everything after the first horizontal corner by one z; None if the
/// path has no corner.
pub fn lifting_path(path: &[Cell]) -> Option<Vec<Cell>> {
    for i in 1..path.len().saturating_sub(1) {
        let (prev, curr, nxt) = (path[i - 1], path[i], path[i + 1]);
        let d1 = (curr.x - prev.x, curr.y - prev.y);
        let d2 = (nxt.x - curr.x, nxt.y - curr.y);
        if d1 != d2 {
            let mut out: Vec<Cell> = path[..=i].to_vec();
            out.push(Cell::new(curr.x, curr.y, curr.z + 1));
            out.extend(path[i + 1..].iter().map(|c| Cell::new(c.x, c.y, c.z + 1)));
            return Some(out);
        }
    }
    None
}

/// Straight vertical path from `pos1` to the z of `pos2`, both ends inclusive.
pub fn vertical_z_path(pos1: Cell, pos2: Cell) -> Vec<Cell> {
    let step: i32 = if pos2.z > pos1.z { 1 } else { -1 };
    let mut out = Vec::new();
    let mut z = pos1.z;
    loop {
        out.push(Cell::new(pos1.x, pos1.y, z));
        if z == pos2.z {
            break;
        }
        z += step;
    }
    out
}

/// Route from `start` to the ceiling cell `target` without rising above
/// `ceiling_z`; None if `target` is occupied or unreachable.
pub fn route_to_ceiling(start: Cell, occ: &Occ, target: Cell, z_floor: f64, ceiling_z: f64, floors: Floors) -> Option<Vec<Cell>> {
    if occ.contains(&target) {
        return None;
    }
    let mut occ_tmp = occ.clone();
    occ_tmp.remove(&start);
    occ_tmp.remove(&target);
    shortest_path(start, target, &occ_tmp, z_floor, floors, &[], Some(ceiling_z), None)
}

/// Result of `route_single_T_to_boundary`.
pub struct TExit {
    pub target: Option<Cell>,
    pub path: Option<Vec<Cell>>,
    /// Orientation after routing: None (Python `0`) once an exit was routed.
    pub ori: Option<Axis>,
}

/// Route a T gate's exit to the nearest side of the footprint (one cell
/// outside the floors). On success the path interior is added to `occ`.
pub fn route_single_t_to_boundary(
    exit: Cell,
    occ: &mut Occ,
    occ_ceiling: &Occ,
    z_floor: f64,
    ceiling_z: f64,
    floors: Floors,
    ori: Option<Axis>,
    region_size: i32,
    idle_cells: &[Cell],
) -> TExit {
    let (x_min, x_max) = (floors.x_min.unwrap() - 1.0, floors.x_max.unwrap() + 1.0);
    let (y_min, y_max) = (floors.y_min.unwrap() - 1.0, floors.y_max.unwrap() + 1.0);
    let (x0, y0, z0) = (exit.x, exit.y, exit.z);
    let fx0 = x0 as f64;
    let fy0 = y0 as f64;

    // nearest boundary; ties resolved like Python's sort of (dist, name)
    let mut dists = [
        ((fx0 - x_min).abs(), "x_min"),
        ((fx0 - x_max).abs(), "x_max"),
        ((fy0 - y_min).abs(), "y_min"),
        ((fy0 - y_max).abs(), "y_max"),
    ];
    dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(b.1)));
    let nearest = dists[0].1;

    let half = region_size / 2;
    let mut region: Vec<Cell> = Vec::new();
    // boundary coordinates are half-integers in Python (x_min_floor - 1); the
    // region cells are Python floats there, but every later use compares or
    // routes them as grid cells, so they are integral in practice.
    let bx = |v: f64| v as i32;
    match nearest {
        "x_min" => {
            for dy in -half..=half {
                for dz in -half..=half {
                    region.push(Cell::new(bx(x_min), y0 + dy, z0 + dz));
                }
            }
        }
        "x_max" => {
            for dy in -half..=half {
                for dz in -half..=half {
                    region.push(Cell::new(bx(x_max), y0 + dy, z0 + dz));
                }
            }
        }
        "y_min" => {
            for dx in -half..=half {
                for dz in -half..=half {
                    region.push(Cell::new(x0 + dx, bx(y_min), z0 + dz));
                }
            }
        }
        _ => {
            for dx in -half..=half {
                for dz in -half..=half {
                    region.push(Cell::new(x0 + dx, bx(y_max), z0 + dz));
                }
            }
        }
    }

    let mut occ_tmp: Occ = occ.clone();
    occ_tmp.extend(occ_ceiling.iter().copied());
    if let Some(axis) = ori {
        for d in axis.offsets() {
            occ_tmp.insert(exit.add(d));
        }
    }
    occ_tmp.remove(&exit);

    for target in region {
        if occ_tmp.contains(&target) || (target.z as f64) < z_floor {
            continue;
        }
        if let Some(path) = shortest_path(exit, target, &occ_tmp, z_floor, floors, idle_cells, Some(ceiling_z), None) {
            for q in crate::geometry::interior(&path) {
                occ.insert(*q);
            }
            return TExit { target: Some(target), path: Some(path), ori: None };
        }
    }
    TExit { target: None, path: None, ori }
}
