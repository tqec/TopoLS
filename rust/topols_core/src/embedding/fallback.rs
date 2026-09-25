//! `basic_embedding` (fallback.py): deterministic brute-force layout of one
//! layer, used when the search and its retries fail.

use rustc_hash::FxHashSet;

use crate::embedding::hadamard::HTable;
use crate::embedding::node::{NodeId, NodeType};
use crate::embedding::state::{NodeMap, Path, TTrack, Track};
use crate::geometry::{Cell, Floors};
use crate::routing::astar::{shortest_path_base, Occ};
use crate::routing::boundary::{lifting_path, vertical_z_path};
use crate::routing::color::{edge_tracer, ori_map, Axis};

pub struct BruteInput<'a> {
    pub pos: &'a NodeMap<Cell>,
    pub ori: &'a NodeMap<Axis>,
    pub typ: &'a NodeMap<NodeType>,
    pub paths: &'a [Path],
    pub occupied: &'a Occ,
    pub z_floor: f64,
    pub floors: Floors,
    pub idle_h_track: &'a NodeMap<Track>,
    pub t_track: &'a NodeMap<TTrack>,
    pub node_type: &'a NodeMap<NodeType>,
    pub input_connect: &'a NodeMap<Vec<NodeId>>,
    pub inter_connect: &'a [(NodeId, NodeId)],
    pub htable: &'a HTable,
}

pub struct BruteOutput {
    pub pos: NodeMap<Cell>,
    pub ori: NodeMap<Axis>,
    pub typ: NodeMap<NodeType>,
    pub paths: Vec<Path>,
    pub occupied: Occ,
    pub idle_h_track: NodeMap<Track>,
    pub idle_place: NodeMap<Cell>,
    pub t_track: NodeMap<TTrack>,
}

fn swap_ij(a: Axis) -> Axis {
    match a {
        Axis::I => Axis::J,
        Axis::J => Axis::I,
        Axis::K => Axis::K,
    }
}

fn side_targets(p: (i32, i32), ori: Axis) -> Vec<(i32, i32)> {
    match ori {
        Axis::I => vec![(p.0 + 1, p.1), (p.0 - 1, p.1)],
        Axis::J => vec![(p.0, p.1 + 1), (p.0, p.1 - 1)],
        Axis::K => vec![],
    }
}

fn reversed(p: &[Cell]) -> Path {
    p.iter().rev().copied().collect()
}

/// Embed one layer without search (see fallback.py for the three steps).
pub fn basic_embedding(inp: &BruteInput) -> BruteOutput {
    let mut pos = inp.pos.clone();
    let mut ori = inp.ori.clone();
    let mut typ = inp.typ.clone();
    let mut paths: Vec<Path> = inp.paths.to_vec();
    let mut occupied: Occ = inp.occupied.clone();
    let mut idle_h_track = inp.idle_h_track.clone();
    let mut idle_place: NodeMap<Cell> = NodeMap::new();
    let mut t_track = inp.t_track.clone();
    let node_type = inp.node_type;
    let input_connect = inp.input_connect;

    let (x_min, x_max) = (inp.floors.x_min.unwrap() - 1.0, inp.floors.x_max.unwrap() + 1.0);
    let (y_min, y_max) = (inp.floors.y_min.unwrap() - 1.0, inp.floors.y_max.unwrap() + 1.0);

    let firsts: Vec<NodeId> = input_connect.values().map(|v| v[0]).collect();
    let mut qubit_pose: NodeMap<Cell> = NodeMap::new();
    for (k, c) in pos.iter() {
        if firsts.contains(k) {
            qubit_pose.insert(*k, *c);
        }
    }
    let wall: FxHashSet<(i32, i32)> = qubit_pose.values().map(|c| (c.x, c.y)).collect();

    // effective ("blue") orientation of every input port
    let mut qubit_ori: NodeMap<Axis> = NodeMap::new();
    let pos_keys: Vec<NodeId> = pos.keys().copied().collect();
    for key in pos_keys {
        if !firsts.contains(&key) {
            continue;
        }
        if let Some(&o) = ori.get(&key) {
            qubit_ori.insert(key, if typ[&key] == 1 { swap_ij(o) } else { o });
        } else {
            let tr = &idle_h_track[&key];
            let st: u8 = match typ[&tr.origin] {
                4 | 5 => 0,
                t => t as u8,
            };
            let (mut ct, ld) = edge_tracer(&reversed(&tr.path), ori[&tr.origin], st);
            // walk input_connect forward through idle/H nodes to the next real node
            let mut nxt = Some(key);
            let mut seen: Vec<NodeId> = Vec::new();
            while let Some(n) = nxt {
                if seen.contains(&n) {
                    break;
                }
                seen.push(n);
                let succ: Vec<NodeId> = input_connect.iter().filter(|(_, v)| !v.is_empty() && v[0] == n).map(|(k, _)| *k).collect();
                if succ.is_empty() {
                    nxt = None;
                    break;
                }
                nxt = Some(succ[0]);
                if !matches!(node_type.get(&succ[0]), Some(2) | Some(3)) {
                    break;
                }
            }
            let target = match nxt {
                Some(n) if !matches!(node_type.get(&n), Some(2) | Some(3)) => n,
                _ => key,
            };
            if inp.htable.needs_flip(tr.origin, target) {
                ct = 1 - ct;
            }
            let o = if ori_map(ld, ct, 0) == Axis::K {
                let red = ori_map(ld, ct, 1);
                if red == Axis::J { Axis::I } else { Axis::J }
            } else {
                ori_map(ld, ct, 0)
            };
            qubit_ori.insert(key, o);
        }
    }

    let z_base = qubit_pose.values().map(|c| c.z).min().unwrap();

    // 1. intra-layer edges (CNOT pairs)
    for &(node1, node2) in inp.inter_connect {
        let (in1, in2) = (input_connect[&node1][0], input_connect[&node2][0]);
        idle_h_track.shift_remove(&in1);
        idle_h_track.shift_remove(&in2);
        let p1 = (qubit_pose[&in1].x, qubit_pose[&in1].y);
        let p2 = (qubit_pose[&in2].x, qubit_pose[&in2].y);
        let o1b = qubit_ori[&in1];
        let o1 = if node_type[&node1] == 1 { o1b } else { swap_ij(o1b) };
        let o2b = qubit_ori[&in2];
        let o2 = if node_type[&node2] == 1 { o2b } else { swap_ij(o2b) };
        let t1s = side_targets(p1, o1);
        let t2s = side_targets(p2, o2);
        let mut z_search = z_base + 1;
        'search: loop {
            for &t1 in &t1s {
                for &t2 in &t2s {
                    if t1 == t2 {
                        continue;
                    }
                    let Some(path_1) = shortest_path_base(Cell::new(t1.0, t1.1, 0), Cell::new(t2.0, t2.1, 0), &occupied, &wall, z_search, inp.floors, None) else { continue };
                    let mut tol: Path = vec![Cell::new(p1.0, p1.1, z_search)];
                    tol.extend_from_slice(&path_1);
                    tol.push(Cell::new(p2.0, p2.1, z_search));
                    if tol.iter().all(|pt| !occupied.contains(&Cell::new(pt.x, pt.y, z_search + 1))) {
                        let Some(tol) = lifting_path(&tol) else { continue };
                        let (n1o, n2o) = (node1.as_old(), node2.as_old());
                        pos.insert(n1o, tol[0]);
                        pos.insert(n2o, *tol.last().unwrap());
                        typ.insert(n1o, node_type[&node1]);
                        typ.insert(n2o, node_type[&node2]);
                        ori.insert(n1o, if node_type[&node1] == 1 { swap_ij(o1b) } else { o1b });
                        ori.insert(n2o, if node_type[&node2] == 1 { swap_ij(o2b) } else { o2b });
                        let p1v = vertical_z_path(qubit_pose[&in1], tol[0]);
                        let p2v = vertical_z_path(qubit_pose[&in2], *tol.last().unwrap());
                        occupied.extend(tol.iter().copied());
                        occupied.extend(p1v.iter().copied());
                        occupied.extend(p2v.iter().copied());
                        paths.push(tol);
                        paths.push(p1v);
                        paths.push(p2v);
                        break 'search;
                    }
                }
            }
            z_search += 1;
        }
    }

    // 2. S and T nodes
    let nodes: Vec<NodeId> = input_connect.keys().copied().collect();
    for &node in &nodes {
        if pos.contains_key(&node) {
            continue;
        }
        let nt = node_type[&node];
        if nt == 4 {
            let inn = input_connect[&node][0];
            idle_h_track.shift_remove(&inn);
            let p = (qubit_pose[&inn].x, qubit_pose[&inn].y);
            let ob = qubit_ori[&inn];
            let targets = side_targets(p, swap_ij(ob));
            let mut z_search = z_base + 1;
            'found: loop {
                for &t in &targets {
                    if !occupied.contains(&Cell::new(t.0, t.1, z_search)) && !occupied.contains(&Cell::new(t.0, t.1, z_search + 1)) {
                        let no = node.as_old();
                        pos.insert(no, Cell::new(p.0, p.1, z_search));
                        typ.insert(no, nt);
                        ori.insert(no, ob);
                        let path = vec![Cell::new(p.0, p.1, z_search), Cell::new(t.0, t.1, z_search), Cell::new(t.0, t.1, z_search + 1)];
                        let pv = vertical_z_path(qubit_pose[&inn], Cell::new(p.0, p.1, z_search));
                        occupied.extend(path.iter().copied());
                        occupied.extend(pv.iter().copied());
                        paths.push(path);
                        paths.push(pv);
                        break 'found;
                    }
                }
                z_search += 1;
            }
        }
        if nt == 5 {
            let inn = input_connect[&node][0];
            idle_h_track.shift_remove(&inn);
            let p = (qubit_pose[&inn].x, qubit_pose[&inn].y);
            let ob = qubit_ori[&inn];
            let targets = side_targets(p, swap_ij(ob));
            let mut z_search = z_base + 1;
            'found_t: loop {
                for &t in &targets {
                    let (fx, fy) = (t.0 as f64, t.1 as f64);
                    let mut dists = [((fx - x_min).abs(), "x_min"), ((fx - x_max).abs(), "x_max"), ((fy - y_min).abs(), "y_min"), ((fy - y_max).abs(), "y_max")];
                    dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(b.1)));
                    let out_target = match dists[0].1 {
                        "x_min" => Cell::new(x_min as i32, t.1, z_search),
                        "x_max" => Cell::new(x_max as i32, t.1, z_search),
                        "y_min" => Cell::new(t.0, y_min as i32, z_search),
                        _ => Cell::new(t.0, y_max as i32, z_search),
                    };
                    let Some(path) = shortest_path_base(Cell::new(t.0, t.1, 0), out_target, &occupied, &wall, z_search, inp.floors, None) else { continue };
                    let mut tol: Path = vec![Cell::new(p.0, p.1, z_search)];
                    tol.extend_from_slice(&path);
                    let pv = vertical_z_path(qubit_pose[&inn], Cell::new(p.0, p.1, z_search));
                    let no = node.as_old();
                    pos.insert(no, tol[0]);
                    typ.insert(no, nt);
                    ori.insert(no, ob);
                    occupied.extend(tol.iter().copied());
                    occupied.extend(pv.iter().copied());
                    t_track.insert(no, TTrack { exit: out_target, path: tol.clone(), ori: None });
                    paths.push(tol);
                    paths.push(pv);
                    break 'found_t;
                }
                z_search += 1;
            }
        }
    }

    // 3. idles and Hadamard boxes on the ceiling
    let z_ceil = pos.values().map(|c| c.z).max().unwrap() + 2;
    let z_layer = if nodes.iter().any(|n| matches!(node_type[n], 4 | 5)) { z_ceil } else { z_ceil - 1 };
    for &node in &nodes {
        if pos.contains_key(&node) {
            continue;
        }
        let nt = node_type[&node];
        if nt == 2 || nt == 3 {
            let inn = input_connect[&node][0];
            let p = (qubit_pose[&inn].x, qubit_pose[&inn].y);
            let at = Cell::new(p.0, p.1, z_layer);
            pos.insert(node, at);
            typ.insert(node, nt);
            let path = vertical_z_path(qubit_pose[&inn], at);
            occupied.extend(path.iter().copied());
            paths.push(path.clone());
            let inc = if nt == 3 { 1 } else { 0 };
            if let Some(tr) = idle_h_track.get(&inn).cloned() {
                let mut np = reversed(&path);
                np.extend_from_slice(&tr.path[1..]);
                idle_h_track.insert(node, Track { origin: tr.origin, path: np, h: tr.h + inc });
                idle_h_track.shift_remove(&inn);
            } else {
                idle_h_track.insert(node, Track { origin: inn, path: reversed(&path), h: inc });
            }
            if nt == 2 {
                idle_place.insert(node, at);
            }
        }
    }

    // stubs above every real node placed as `_old`
    for &node in &nodes {
        if !pos.contains_key(&node) {
            let old = pos[&node.as_old()];
            let at = Cell::new(old.x, old.y, z_ceil);
            pos.insert(node, at);
            typ.insert(node, 2);
            let path = vertical_z_path(old, at);
            occupied.extend(path.iter().copied());
            paths.push(path.clone());
            idle_h_track.insert(node, Track { origin: node.as_old(), path: reversed(&path), h: 0 });
            idle_place.insert(node, at);
        }
    }

    BruteOutput { pos, ori, typ, paths, occupied, idle_h_track, idle_place, t_track }
}
