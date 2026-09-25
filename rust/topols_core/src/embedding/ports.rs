//! Input-port layout, the lift of a finished layer to a common ceiling and
//! the final seals (ports.py).

use indexmap::IndexMap;

use crate::embedding::node::{is_cube, trace_type, NodeId, NodeType};
use crate::embedding::state::{CeilingEntry, EmbeddingState, NodeMap, Path, Track};
use crate::geometry::{Cell, Floors};
use crate::routing::astar::Occ;
use crate::routing::boundary::vertical_z_path;
use crate::routing::color::{edge_tracer, ori_map, Axis};

/// `auto_ports`: qubit i -> cell on a serpentine grid at `z_level`, `length`
/// per row, `edge_dist` apart. Orientation 'i' and type 0 for every port.
pub fn auto_ports(num_qubits: usize, z_level: i32, edge_dist: i32, length: Option<usize>) -> IndexMap<usize, Cell> {
    let length = length.unwrap_or_else(|| (num_qubits as f64).sqrt().ceil() as usize);
    let width = (num_qubits + length - 1) / length;
    let mut loc = IndexMap::new();
    let mut idx = 0usize;
    'rows: for j in 0..width {
        let row: Vec<usize> = if j % 2 == 0 { (0..length).collect() } else { (0..length).rev().collect() };
        for i in row {
            if idx >= num_qubits {
                break 'rows;
            }
            loc.insert(idx, Cell::new((i as i32) * edge_dist, (j as i32) * edge_dist, z_level));
            idx += 1;
        }
        if idx >= num_qubits {
            break;
        }
    }
    loc
}

pub const PORT_ORI: Axis = Axis::I;

/// `calculate_space_time`: (x_length, y_length, z_length, volume).
pub fn calculate_space_time(pos: &NodeMap<Cell>, paths: &[Path], floors: &Floors) -> (f64, f64, f64, f64) {
    let zs = pos.values().map(|c| c.z).chain(paths.iter().flat_map(|p| p.iter().map(|c| c.z)));
    let (mut zmin, mut zmax) = (i32::MAX, i32::MIN);
    for z in zs {
        zmin = zmin.min(z);
        zmax = zmax.max(z);
    }
    let x_length = floors.x_max.unwrap() - floors.x_min.unwrap() + 1.0;
    let y_length = floors.y_max.unwrap() - floors.y_min.unwrap() + 1.0;
    let z_length = (zmax - zmin) as f64;
    (x_length, y_length, z_length, x_length * y_length * z_length)
}

fn rename_old<V: Clone>(map: &mut NodeMap<V>, node_type: &NodeMap<NodeType>) {
    let keys: Vec<NodeId> = map.keys().copied().collect();
    for key in keys {
        if node_type.contains_key(&key) {
            let v = map[&key].clone();
            map.insert(key.as_old(), v);
            map.shift_remove(&key);
        }
    }
}

fn reversed(p: &[Cell]) -> Path {
    p.iter().rev().copied().collect()
}

/// `ceiling`: commit the lift described by `ceiling_track` to `state`
/// (mutated and returned). With `final_seal`, colour every output end.
pub fn ceiling(state: &mut EmbeddingState, ceiling_track: &mut NodeMap<CeilingEntry>, node_type: &NodeMap<NodeType>, final_seal: bool) {
    let mut occ: Occ = state.occupied.clone();

    rename_old(&mut state.pos, node_type);
    rename_old(&mut state.ori, node_type);
    rename_old(&mut state.typ, node_type);
    rename_old(&mut state.t_track, node_type);

    for (key, e) in ceiling_track.iter() {
        state.pos.insert(*key, *e.path.last().unwrap());
    }
    for (key, e) in ceiling_track.iter() {
        if let Some(o) = e.ori {
            state.ori.insert(*key, o);
        }
    }

    if final_seal {
        // A real node whose lift ends at the output port: if the wire from the
        // node to the port carries an odd number of Hadamards, the end colour
        // computed by reward() must flip.
        for (key, e) in ceiling_track.iter_mut() {
            let Some(&t0) = node_type.get(key) else { continue };
            if !is_cube(t0) || e.ori.is_none() {
                continue;
            }
            let old = key.as_old();
            if !state.ori.contains_key(&old) || !state.htable.needs_flip_to_end(old) {
                continue;
            }
            if t0 == 4 || t0 == 5 {
                let (ct, ld) = edge_tracer(&e.path, state.ori[&old], 0);
                let ct = 1 - ct;
                if ori_map(ld, ct, 0) == Axis::K {
                    e.ori = Some(ori_map(ld, ct, 1));
                    e.typ = 1;
                } else {
                    e.ori = Some(ori_map(ld, ct, 0));
                    e.typ = t0;
                }
            } else {
                let (ct, ld) = edge_tracer(&e.path, state.ori[&old], t0 as u8);
                let ct = 1 - ct;
                if ori_map(ld, ct, t0 as u8) == Axis::K {
                    let t2: u8 = if t0 != 1 { 1 } else { 0 };
                    e.ori = Some(ori_map(ld, ct, t2));
                    e.typ = t2 as NodeType;
                } else {
                    e.ori = Some(ori_map(ld, ct, t0 as u8));
                    e.typ = t0;
                }
            }
            state.ori.insert(*key, e.ori.unwrap());
        }
    }

    for (key, e) in ceiling_track.iter() {
        let mut t = e.typ;
        if t > 1 && t != 4 && t != 5 {
            if final_seal {
                t = 0;
                let tr = state.idle_h_track.get(key).expect("chain end without track").clone();
                let mut tol: Path = reversed(&e.path);
                tol.extend_from_slice(&tr.path[1..]);
                state.idle_h_track.insert(*key, Track { origin: tr.origin, path: tol.clone(), h: tr.h });
                let (mut ct, ld) = edge_tracer(&reversed(&tol), state.ori[&tr.origin], trace_type(state.typ[&tr.origin]));
                if state.htable.needs_flip_to_end(tr.origin) {
                    ct = 1 - ct;
                }
                state.ori.insert(*key, ori_map(ld, ct, 0));
            } else {
                t = 2;
                if let Some(tr) = state.idle_h_track.get(key).cloned() {
                    let mut p: Path = reversed(&e.path);
                    p.extend_from_slice(&tr.path[1..]);
                    state.idle_h_track.insert(*key, Track { origin: tr.origin, path: p, h: tr.h });
                } else {
                    state.idle_h_track.insert(*key, Track { origin: key.as_old(), path: reversed(&e.path), h: 0 });
                }
            }
        } else if t == 4 || t == 5 {
            t = 0;
        }
        state.typ.insert(*key, t);
    }

    for e in ceiling_track.values() {
        state.paths.push(e.path.clone());
        occ.extend(e.path.iter().copied());
    }
    state.occupied = occ;

    let mut idle_place = NodeMap::new();
    for (key, e) in ceiling_track.iter() {
        if e.typ > 1 && e.typ != 4 && e.typ != 5 {
            idle_place.insert(*key, *e.path.last().unwrap());
        }
    }
    state.idle_place = idle_place;
}

/// `seal_brute_frontier`: final seal of a frontier left by `basic_embedding`.
pub fn seal_brute_frontier(state: &mut EmbeddingState) {
    let frontier: Vec<NodeId> = state
        .idle_h_track
        .iter()
        .filter(|(k, tr)| matches!(state.typ.get(*k), Some(2) | Some(3)) && state.ori.contains_key(&tr.origin))
        .map(|(k, _)| *k)
        .collect();
    if frontier.is_empty() {
        return;
    }
    let mut occ = state.occupied.clone();
    let z_top = frontier.iter().map(|k| state.pos[k].z).max().unwrap();
    let mut paths = state.paths.clone();
    for &key in &frontier {
        let Cell { x, y, z } = state.pos[&key];
        if z >= z_top {
            continue;
        }
        if (z + 1..=z_top).any(|zz| occ.contains(&Cell::new(x, y, zz))) {
            continue;
        }
        let seg = vertical_z_path(Cell::new(x, y, z), Cell::new(x, y, z_top));
        let tr = state.idle_h_track[&key].clone();
        let mut p: Path = reversed(&seg);
        p.extend_from_slice(&tr.path[1..]);
        state.idle_h_track.insert(key, Track { origin: tr.origin, path: p, h: tr.h });
        state.pos.insert(key, Cell::new(x, y, z_top));
        if state.idle_place.contains_key(&key) {
            state.idle_place.insert(key, Cell::new(x, y, z_top));
        }
        let old_end = Cell::new(x, y, z);
        let mut extended = false;
        for pth in paths.iter_mut() {
            if *pth.last().unwrap() == old_end {
                pth.extend_from_slice(&seg[1..]);
                extended = true;
                break;
            }
            if pth[0] == old_end {
                let mut np: Path = seg[1..].iter().rev().copied().collect();
                np.extend_from_slice(pth);
                *pth = np;
                extended = true;
                break;
            }
        }
        if !extended {
            paths.push(seg.clone());
        }
        occ.extend(seg.iter().copied());
    }
    state.paths = paths;
    state.occupied = occ;

    for &key in &frontier {
        let tr = state.idle_h_track[&key].clone();
        let st = match state.typ.get(&tr.origin) {
            Some(4) | Some(5) => 0u8,
            Some(t) => *t as u8,
            None => 0,
        };
        let (mut ct, ld) = edge_tracer(&reversed(&tr.path), state.ori[&tr.origin], st);
        if state.htable.needs_flip_to_end(tr.origin) {
            ct = 1 - ct;
        }
        state.ori.insert(key, ori_map(ld, ct, 0));
        state.typ.insert(key, 0);
    }
}
