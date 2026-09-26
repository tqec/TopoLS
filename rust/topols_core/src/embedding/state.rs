//! `EmbeddingState` (state.py): a partial embedding of one layer, the MCTS
//! search state. `next_state` places one node and routes its wires;
//! `reward` completes a terminal state.
//!
//! Conventions follow the Python implementation exactly: dicts are
//! insertion-ordered maps, the intra-layer edges are iterated in
//! `ordered_edges` order, and every routing decision is made in the same
//! order with the same tie-breaks. Two representation choices differ from
//! Python without changing any result: temporary occupancies are `OccView`s
//! over the shared set instead of copies, and the routed paths are a
//! persistent vector (O(1) clone) with the highest path z carried along.

use std::sync::Arc;

use im::Vector;
use indexmap::IndexMap;
use rustc_hash::FxHashSet;

use crate::embedding::hadamard::HTable;
use crate::embedding::node::{is_chain, is_cube, trace_type, NodeId, NodeType};
use crate::embedding::ports::auto_ports;
use crate::geometry::{interior, Cell, Floors};
use crate::pyrandom::PyRandom;
use crate::routing::astar::{add_work, shortest_path, Blocked, Occ, OccView, NEXT_STATE_COST};
use crate::routing::boundary::{route_single_t_to_boundary, route_to_ceiling};
use crate::routing::color::{color_switch, edge_tracer, ori_map, Axis};

pub type Path = Vec<Cell>;
pub type NodeMap<V> = IndexMap<NodeId, V>;
/// Routed paths of a state; persistent so that a successor shares storage.
pub type Paths = Vector<Path>;

/// `idle_h_track[node] = [origin, path, h]`: for the idle at the end of a
/// chain, the real node the chain started from and the path from the chain
/// end back to that origin.
#[derive(Clone, Debug)]
pub struct Track {
    pub origin: NodeId,
    pub path: Path,
    pub h: i32,
}

/// `t_track[node] = [exit, path, ori]`: the exit stub of an S/T node.
#[derive(Clone, Debug)]
pub struct TTrack {
    pub exit: Cell,
    pub path: Path,
    /// Python `ori`: an axis while the exit is the node itself, `0` (None)
    /// once routed.
    pub ori: Option<Axis>,
}

/// The layer being embedded (`layer_info` output), shared by all states of a
/// search.
#[derive(Clone, Debug, Default)]
pub struct Layer {
    pub node_type: NodeMap<NodeType>,
    pub inter_connect: Vec<(NodeId, NodeId)>,
    pub output_connect: NodeMap<i32>,
}

/// One lifted end in `ceiling_track`.
#[derive(Clone, Debug)]
pub struct CeilingEntry {
    pub path: Path,
    pub ori: Option<Axis>,
    pub typ: NodeType,
}

pub struct RewardResult {
    pub reward: f64,
    pub t_track: NodeMap<TTrack>,
    pub occupied: Occ,
    pub ceiling_track: NodeMap<CeilingEntry>,
}

#[derive(Clone)]
pub struct EmbeddingState {
    pub pos: NodeMap<Cell>,
    pub ori: NodeMap<Axis>,
    pub typ: NodeMap<NodeType>,
    pub paths: Paths,
    /// Highest z of any cell in `paths` (i32::MIN when there are none).
    pub paths_max_z: i32,
    pub occupied: Occ,
    pub z_floor: f64,
    pub floors: Floors,
    pub idle_h_track: NodeMap<Track>,
    pub idle_place: NodeMap<Cell>,
    pub t_track: NodeMap<TTrack>,
    pub layer: Arc<Layer>,
    /// Per-seed shuffled input ports (`input_connect`).
    pub input_connect: Arc<NodeMap<Vec<NodeId>>>,
    pub order: Arc<Vec<NodeId>>,
    pub z_length: f64,
    pub htable: Arc<HTable>,
    pub order_idx: usize,
    pub vol: f64,
}

/// Highest z over the cells of `paths`.
pub fn paths_max_z<'a>(paths: impl IntoIterator<Item = &'a Path>) -> i32 {
    paths.into_iter().flat_map(|p| p.iter().map(|c| c.z)).max().unwrap_or(i32::MIN)
}

/// `geometry.bounding_box`
pub fn bounding_box(pos: &NodeMap<Cell>, paths_max_z: i32, floors: &Floors, min_z: f64, z_length: f64) -> f64 {
    let max_z = pos.values().map(|c| c.z).max().unwrap().max(paths_max_z);
    (floors.x_max.unwrap() - floors.x_min.unwrap() + 1.0)
        * (floors.y_max.unwrap() - floors.y_min.unwrap() + 1.0)
        * (max_z as f64 - min_z + z_length)
}

#[inline]
fn idle_cells_masked(idle_place: &NodeMap<Cell>, mask: Option<NodeId>) -> Vec<Cell> {
    idle_place.iter().filter(|(k, _)| Some(**k) != mask).map(|(_, v)| *v).collect()
}

#[inline]
fn push_offsets(extra: &mut Vec<Cell>, at: Cell, axis: Axis) {
    for d in axis.offsets() {
        extra.push(at.add(d));
    }
}

fn reversed(path: &[Cell]) -> Vec<Cell> {
    path.iter().rev().copied().collect()
}

/// `_hadamard_flip`: colour type of a wire from real node `a` to real node `b`.
#[inline]
fn hadamard_flip(ht: &HTable, t: u8, a: NodeId, b: NodeId) -> u8 {
    if ht.needs_flip(a, b) { 1 - t } else { t }
}

/// Trace a chain from its origin along `tol` (origin ... end), reversed as
/// Python does: `edge_tracer(tol[::-1], (ori[origin], trace_type))`.
fn trace_from_origin(ori: &NodeMap<Axis>, typ: &NodeMap<NodeType>, origin: NodeId, tol: &[Cell]) -> (u8, Axis) {
    edge_tracer(&reversed(tol), ori[&origin], trace_type(typ[&origin]))
}

/// Working copies of a state's mutable parts while a placement is routed.
struct Work {
    pos: NodeMap<Cell>,
    ori: NodeMap<Axis>,
    typ: NodeMap<NodeType>,
    paths: Paths,
    paths_max_z: i32,
    occ: Occ,
    track: NodeMap<Track>,
    t_track: NodeMap<TTrack>,
    idle_place: NodeMap<Cell>,
}

impl Work {
    fn push_path(&mut self, path: Path) {
        for c in &path {
            if c.z > self.paths_max_z {
                self.paths_max_z = c.z;
            }
        }
        self.paths.push_back(path);
    }
}

/// The temporary occupancy of a routing helper: `occ - removed + extra`.
struct Temp {
    removed: Vec<Cell>,
    extra: Vec<Cell>,
}
impl Temp {
    fn view<'a>(&'a self, base: &'a Occ) -> OccView<'a> {
        OccView { base, removed: &self.removed, extra: &self.extra }
    }
}

#[allow(clippy::too_many_arguments)]
impl EmbeddingState {
    pub fn new(
        pos: NodeMap<Cell>,
        ori: NodeMap<Axis>,
        typ: NodeMap<NodeType>,
        paths: Paths,
        paths_max_z: i32,
        occupied: Occ,
        z_floor: f64,
        floors: Floors,
        idle_h_track: NodeMap<Track>,
        idle_place: NodeMap<Cell>,
        t_track: NodeMap<TTrack>,
        layer: Arc<Layer>,
        input_connect: Arc<NodeMap<Vec<NodeId>>>,
        order: Arc<Vec<NodeId>>,
        z_length: f64,
        htable: Arc<HTable>,
        order_idx: usize,
    ) -> EmbeddingState {
        let vol = if pos.len() < 2 { 0.0 } else { bounding_box(&pos, paths_max_z, &floors, z_floor, z_length) };
        EmbeddingState { pos, ori, typ, paths, paths_max_z, occupied, z_floor, floors, idle_h_track, idle_place, t_track, layer, input_connect, order, z_length, htable, order_idx, vol }
    }

    pub fn is_terminal(&self) -> bool {
        self.order_idx >= self.order.len()
    }

    /// `moves`: candidate cells for the next node. Consumes the RNG exactly
    /// where Python's `random.shuffle` does.
    pub fn moves(&self, rng: &mut PyRandom, num: usize, block_switch: bool, ceiling_switch: bool, rollout: bool) -> Vec<Cell> {
        if self.is_terminal() {
            return vec![];
        }
        let node = self.order[self.order_idx];
        let input0 = self.input_connect[&node][0];
        let cent = self.pos[&input0];
        let up = cent.add(Cell::new(0, 0, 1));
        let nt = self.layer.node_type[&node];
        if (self.typ[&input0] == 2 && nt == 2) || (block_switch && nt == 2) || (ceiling_switch && nt == 2) {
            return vec![up];
        }
        let pmove_1: &[Cell] = if num == 1 || rollout {
            &[Cell::new(0, 0, 1)]
        } else {
            &[Cell::new(1, 0, 0), Cell::new(-1, 0, 0), Cell::new(0, 1, 0), Cell::new(0, -1, 0), Cell::new(0, 0, 1), Cell::new(0, 0, -1)]
        };
        let mut cand_1: Vec<Cell> = Vec::new();
        for &pm in pmove_1 {
            let p = cent.add(pm);
            if self.occupied.contains(&p) || (p.z as f64) < self.z_floor {
                continue;
            }
            cand_1.push(p);
        }
        if cand_1.len() >= num {
            rng.shuffle(&mut cand_1);
            cand_1.truncate(num);
            return cand_1;
        }
        let pmove_2 = [
            Cell::new(1, 1, 1), Cell::new(-1, 1, 1), Cell::new(1, -1, 1), Cell::new(-1, -1, 1),
            Cell::new(1, 0, 1), Cell::new(-1, 0, 1), Cell::new(0, 1, 1), Cell::new(0, -1, 1),
        ];
        let mut cand_2: Vec<Cell> = Vec::new();
        for pm in pmove_2 {
            let p = cent.add(pm);
            if self.occupied.contains(&p) || (p.z as f64) < self.z_floor {
                continue;
            }
            cand_2.push(p);
        }
        if cand_1.len() + cand_2.len() >= num {
            rng.shuffle(&mut cand_2);
            cand_2.truncate(num - cand_1.len());
            cand_1.extend(cand_2);
            return cand_1;
        }
        cand_1.extend(cand_2);
        cand_1
    }

    // ------------------------------------------------------------------
    // routing helpers (module functions in Python)
    // ------------------------------------------------------------------

    /// `_route_input_ports`. Returns the temporary occupancy of the last port
    /// routed (the S branch extends it) and that port, or None.
    fn route_input_ports(&self, w: &mut Work, node: NodeId, coord: Cell, input_ports: &[NodeId], target_type: u8) -> Option<(Path, Temp, NodeId)> {
        let ht = &*self.htable;
        let mut ori_flag = false;
        let mut last: Option<(Path, Temp, NodeId)> = None;
        for &input in input_ports {
            // occ_tmp = occ - {pos[input], coord} (+ offsets around a cube port)
            debug_assert!(w.occ.contains(&w.pos[&input]) && w.occ.contains(&coord));
            let mut tmp = Temp { removed: vec![w.pos[&input], coord], extra: Vec::new() };
            if is_cube(w.typ[&input]) {
                push_offsets(&mut tmp.extra, w.pos[&input], w.ori[&input]);
                if tmp.view(&w.occ).contains(&coord) {
                    return None;
                }
            }
            let idle_cells = idle_cells_masked(&w.idle_place, Some(input));
            let mut path;
            if !ori_flag {
                path = shortest_path(coord, w.pos[&input], &tmp.view(&w.occ), self.z_floor, self.floors, &idle_cells, None, None)?;
                let ti = w.typ[&input];
                if ti == 0 || ti == 1 {
                    let (ct, ld) = edge_tracer(&reversed(&path), w.ori[&input], ti as u8);
                    let ct = hadamard_flip(ht, ct, node, input);
                    w.ori.insert(node, ori_map(ld, ct, target_type));
                } else if is_chain(ti) {
                    let tr = &w.track[&input];
                    let mut tol = path.clone();
                    tol.extend_from_slice(&tr.path[1..]);
                    let (mut ct, ld) = trace_from_origin(&w.ori, &w.typ, tr.origin, &tol);
                    if ht.needs_flip(tr.origin, node) {
                        ct = 1 - ct;
                    }
                    w.ori.insert(node, ori_map(ld, ct, target_type));
                    w.track.shift_remove(&input);
                } else {
                    let (ct, ld) = edge_tracer(&reversed(&path), w.ori[&input], 0);
                    let ct = hadamard_flip(ht, ct, node, input);
                    w.ori.insert(node, ori_map(ld, ct, target_type));
                }
                ori_flag = true;
            } else {
                push_offsets(&mut tmp.extra, w.pos[&node], w.ori[&node]);
                path = shortest_path(coord, w.pos[&input], &tmp.view(&w.occ), self.z_floor, self.floors, &idle_cells, None, None)?;
                let ti = w.typ[&input];
                let (ct, ld) = if ti == 0 || ti == 1 {
                    let (ct, ld) = edge_tracer(&reversed(&path), w.ori[&input], ti as u8);
                    (hadamard_flip(ht, ct, node, input), ld)
                } else if is_chain(ti) {
                    let tr = &w.track[&input];
                    let mut tol = path.clone();
                    tol.extend_from_slice(&tr.path[1..]);
                    let (mut ct, ld) = trace_from_origin(&w.ori, &w.typ, tr.origin, &tol);
                    if ht.needs_flip(tr.origin, node) {
                        ct = 1 - ct;
                    }
                    w.track.shift_remove(&input);
                    (ct, ld)
                } else {
                    let (ct, ld) = edge_tracer(&reversed(&path), w.ori[&input], 0);
                    (hadamard_flip(ht, ct, node, input), ld)
                };
                if w.ori[&node] != ori_map(ld, ct, target_type) {
                    path = color_switch(&path, &tmp.view(&w.occ), self.z_floor, self.floors)?;
                }
            }
            w.occ.extend(interior(&path).iter().copied());
            w.push_path(path.clone());
            if w.typ[&input] == 2 {
                w.idle_place.shift_remove(&input).expect("idle input not in idle_place");
            }
            last = Some((path, tmp, input));
        }
        last
    }

    /// `_route_solid_src_to_solid_dst`
    fn route_solid_src_to_solid_dst(&self, w: &mut Work, src: NodeId, dst: NodeId, dst_typ: NodeType, typ_input: u8, mask: NodeId) -> Option<Path> {
        let (sc, dc) = (w.pos[&src], w.pos[&dst]);
        let mut tmp = Temp { removed: vec![sc, dc], extra: Vec::new() };
        push_offsets(&mut tmp.extra, sc, w.ori[&src]);
        push_offsets(&mut tmp.extra, dc, w.ori[&dst]);
        let view = tmp.view(&w.occ);
        if view.contains(&sc) || view.contains(&dc) {
            return None;
        }
        let idle_cells = idle_cells_masked(&w.idle_place, Some(mask));
        let mut path = shortest_path(dc, sc, &view, self.z_floor, self.floors, &idle_cells, None, None)?;
        let (ct, ld) = edge_tracer(&reversed(&path), w.ori[&src], typ_input);
        let ct = hadamard_flip(&self.htable, ct, src, dst);
        if w.ori[&dst] != ori_map(ld, ct, if dst_typ == 1 { 1 } else { 0 }) {
            path = color_switch(&path, &view, self.z_floor, self.floors)?;
        }
        w.occ.extend(interior(&path).iter().copied());
        Some(path)
    }

    /// `_route_chain_src_to_solid_dst`
    fn route_chain_src_to_solid_dst(&self, w: &mut Work, src: NodeId, dst: NodeId, dst_typ: NodeType, mask: NodeId) -> Option<Path> {
        let (sc, dc) = (w.pos[&src], w.pos[&dst]);
        let typ_output: u8 = if dst_typ == 1 { 1 } else { 0 };
        let mut tmp = Temp { removed: vec![sc, dc], extra: Vec::new() };
        push_offsets(&mut tmp.extra, dc, w.ori[&dst]);
        let view = tmp.view(&w.occ);
        if view.contains(&sc) || view.contains(&dc) {
            return None;
        }
        let idle_cells = idle_cells_masked(&w.idle_place, Some(mask));
        let mut path = shortest_path(dc, sc, &view, self.z_floor, self.floors, &idle_cells, None, None)?;
        let tr = &w.track[&src];
        let mut tol = path.clone();
        tol.extend_from_slice(&tr.path[1..]);
        let (mut ct, ld) = trace_from_origin(&w.ori, &w.typ, tr.origin, &tol);
        if self.htable.needs_flip(tr.origin, dst) {
            ct = 1 - ct;
        }
        if w.ori[&dst] != ori_map(ld, ct, typ_output) {
            path = color_switch(&path, &view, self.z_floor, self.floors)?;
        }
        w.track.shift_remove(&src);
        w.occ.extend(interior(&path).iter().copied());
        Some(path)
    }

    /// `_route_solid_src_to_chain_dst`
    fn route_solid_src_to_chain_dst(&self, w: &mut Work, src: NodeId, dst: NodeId, target_type: u8, mask: NodeId) -> Option<Path> {
        let (sc, dc) = (w.pos[&src], w.pos[&dst]);
        let mut tmp = Temp { removed: vec![sc, dc], extra: Vec::new() };
        push_offsets(&mut tmp.extra, sc, w.ori[&src]);
        let view = tmp.view(&w.occ);
        if view.contains(&sc) || view.contains(&dc) {
            return None;
        }
        let idle_cells = idle_cells_masked(&w.idle_place, Some(mask));
        let mut path = shortest_path(sc, dc, &view, self.z_floor, self.floors, &idle_cells, None, None)?;
        let tr = &w.track[&dst];
        let mut tol = path.clone();
        tol.extend_from_slice(&tr.path[1..]);
        let (mut ct, ld) = trace_from_origin(&w.ori, &w.typ, tr.origin, &tol);
        if self.htable.needs_flip(src, tr.origin) {
            ct = 1 - ct;
        }
        if w.ori[&src] != ori_map(ld, ct, target_type) {
            path = color_switch(&path, &view, self.z_floor, self.floors)?;
        }
        w.track.shift_remove(&dst);
        w.occ.extend(interior(&path).iter().copied());
        Some(path)
    }

    /// `_route_chain_src_to_chain_dst`
    fn route_chain_src_to_chain_dst(&self, w: &mut Work, src: NodeId, dst: NodeId, mask: NodeId) -> Option<Path> {
        let (sc, dc) = (w.pos[&src], w.pos[&dst]);
        let tmp = Temp { removed: vec![sc, dc], extra: Vec::new() };
        let view = tmp.view(&w.occ);
        if view.contains(&sc) || view.contains(&dc) {
            return None;
        }
        let idle_cells = idle_cells_masked(&w.idle_place, Some(mask));
        let mut path = shortest_path(sc, dc, &view, self.z_floor, self.floors, &idle_cells, None, None)?;
        let (td, ts) = (&w.track[&dst], &w.track[&src]);
        let mut tol: Path = reversed(&ts.path);
        tol.extend_from_slice(&path[1..]);
        tol.extend_from_slice(&td.path[1..]);
        let (mut ct, ld) = trace_from_origin(&w.ori, &w.typ, td.origin, &tol);
        if self.htable.needs_flip(ts.origin, td.origin) {
            ct = 1 - ct;
        }
        if w.ori[&ts.origin] != ori_map(ld, ct, if w.typ[&ts.origin] == 1 { 1 } else { 0 }) {
            path = color_switch(&path, &view, self.z_floor, self.floors)?;
        }
        w.track.shift_remove(&src);
        w.track.shift_remove(&dst);
        w.occ.extend(interior(&path).iter().copied());
        Some(path)
    }

    /// Intra-layer edges of `node` whose other endpoint is already placed,
    /// in `inter_connect` order: `(dst, dst_typ)`.
    fn placed_partners(&self, node: NodeId, w: &Work) -> Vec<(NodeId, NodeType)> {
        let mut out = Vec::new();
        for &(a, b) in &self.layer.inter_connect {
            if (a == node && w.pos.contains_key(&b)) || (b == node && w.pos.contains_key(&a)) {
                let dst = if a == node { b } else { a };
                out.push((dst, w.typ[&dst]));
            }
        }
        out
    }

    /// Route the intra-layer edges of a freshly placed cube (types 0/1/4/5).
    fn route_partners_from_cube(&self, w: &mut Work, node: NodeId, typ_input: u8, mask: NodeId) -> Option<()> {
        for (dst, dst_typ) in self.placed_partners(node, w) {
            let path = if is_cube(dst_typ) {
                self.route_solid_src_to_solid_dst(w, node, dst, dst_typ, typ_input, mask)?
            } else {
                self.route_solid_src_to_chain_dst(w, node, dst, typ_input, mask)?
            };
            w.push_path(path);
        }
        Some(())
    }

    /// Route the intra-layer edges of a freshly placed chain end (types 2/3).
    fn route_partners_from_chain(&self, w: &mut Work, node: NodeId, mask: NodeId) -> Option<()> {
        for (dst, dst_typ) in self.placed_partners(node, w) {
            let path = if is_cube(dst_typ) {
                self.route_chain_src_to_solid_dst(w, node, dst, dst_typ, mask)?
            } else {
                self.route_chain_src_to_chain_dst(w, node, dst, mask)?
            };
            w.push_path(path);
        }
        Some(())
    }

    /// `next_state`: place the next node of `order` at `coord`.
    pub fn next_state(&self, coord: Cell) -> Option<EmbeddingState> {
        add_work(NEXT_STATE_COST);
        let node = self.order[self.order_idx];
        let mut input = self.input_connect[&node][0];

        let mut w = Work {
            pos: self.pos.clone(),
            ori: self.ori.clone(),
            typ: self.typ.clone(),
            paths: self.paths.clone(),
            paths_max_z: self.paths_max_z,
            occ: self.occupied.clone(),
            track: self.idle_h_track.clone(),
            t_track: self.t_track.clone(),
            idle_place: self.idle_place.clone(),
        };

        if w.occ.contains(&coord) || (coord.z as f64) < self.z_floor {
            return None;
        }
        if self.typ[&input] != 2 {
            for c in w.idle_place.values() {
                if coord.x == c.x && coord.y == c.y && coord.z >= c.z {
                    return None;
                }
            }
        }
        let nt = self.layer.node_type[&node];
        w.pos.insert(node, coord);
        w.typ.insert(node, nt);
        w.occ.insert(coord);

        match nt {
            0 | 1 => {
                let (_, _, inp) = self.route_input_ports(&mut w, node, coord, &self.input_connect[&node], nt as u8)?;
                input = inp;
                self.route_partners_from_cube(&mut w, node, nt as u8, input)?;
            }
            2 => {
                let top = w.occ.iter().filter(|&&c| c != coord).map(|c| c.z).max().unwrap();
                if w.typ[&input] == 2 && w.pos[&input].z >= top {
                    // consecutive idles at the top collapse into one cell
                    let at = w.pos[&input];
                    w.pos.insert(node, at);
                    w.occ.remove(&coord);
                    w.idle_place.insert(node, at);
                    w.idle_place.shift_remove(&input);
                    let tr = w.track[&input].clone();
                    w.track.insert(node, tr);
                    w.track.shift_remove(&input);
                } else {
                    let mut tmp = Temp { removed: vec![w.pos[&input], coord], extra: Vec::new() };
                    if is_cube(w.typ[&input]) {
                        push_offsets(&mut tmp.extra, w.pos[&input], w.ori[&input]);
                    }
                    if tmp.view(&w.occ).contains(&coord) {
                        return None;
                    }
                    let idle_cells = idle_cells_masked(&w.idle_place, Some(input));
                    let path = shortest_path(coord, w.pos[&input], &tmp.view(&w.occ), self.z_floor, self.floors, &idle_cells, None, None)?;
                    w.occ.extend(interior(&path).iter().copied());
                    w.push_path(path.clone());
                    if is_cube(w.typ[&input]) {
                        w.track.insert(node, Track { origin: input, path, h: 0 });
                    } else {
                        let tr = &w.track[&input];
                        let mut p = path;
                        p.extend_from_slice(&tr.path[1..]);
                        let new = Track { origin: tr.origin, path: p, h: tr.h };
                        w.track.insert(node, new);
                        w.track.shift_remove(&input);
                    }
                    if w.typ[&input] != 2 {
                        for c in w.occ.iter() {
                            if c.x == coord.x && c.y == coord.y && c.z > coord.z {
                                return None;
                            }
                        }
                        w.idle_place.insert(node, coord);
                    } else {
                        w.idle_place.insert(node, coord);
                        w.idle_place.shift_remove(&input);
                    }
                    self.route_partners_from_chain(&mut w, node, input)?;
                }
            }
            3 => {
                let mut ori_flag = false;
                for &inp in self.input_connect[&node].iter() {
                    input = inp;
                    if w.typ[&inp] == 2 {
                        w.idle_place.shift_remove(&inp).expect("idle input not in idle_place");
                    }
                    let mut tmp = Temp { removed: vec![w.pos[&inp], coord], extra: Vec::new() };
                    if is_cube(w.typ[&inp]) {
                        push_offsets(&mut tmp.extra, w.pos[&inp], w.ori[&inp]);
                        if tmp.view(&w.occ).contains(&coord) {
                            return None;
                        }
                    }
                    if !ori_flag {
                        let idle_cells = idle_cells_masked(&w.idle_place, Some(inp));
                        let path = shortest_path(coord, w.pos[&inp], &tmp.view(&w.occ), self.z_floor, self.floors, &idle_cells, None, None)?;
                        if is_cube(w.typ[&inp]) {
                            w.track.insert(node, Track { origin: inp, path: path.clone(), h: 1 });
                        } else {
                            let tr = &w.track[&inp];
                            let mut p = path.clone();
                            p.extend_from_slice(&tr.path[1..]);
                            let new = Track { origin: tr.origin, path: p, h: tr.h + 1 };
                            w.track.insert(node, new);
                            w.track.shift_remove(&inp);
                        }
                        w.occ.extend(interior(&path).iter().copied());
                        w.push_path(path);
                        ori_flag = true;
                    } else {
                        let dst_typ = w.typ[&inp];
                        let path = if is_cube(dst_typ) {
                            self.route_chain_src_to_solid_dst(&mut w, node, inp, dst_typ, inp)?
                        } else {
                            self.route_chain_src_to_chain_dst(&mut w, node, inp, inp)?
                        };
                        w.push_path(path);
                    }
                }
                self.route_partners_from_chain(&mut w, node, input)?;
            }
            4 => {
                let (path, mut tmp, inp) = self.route_input_ports(&mut w, node, coord, &self.input_connect[&node], 0)?;
                input = inp;
                // occ_tmp.add(pos[input]); occ_tmp.add(coord)
                tmp.extra.push(w.pos[&input]);
                tmp.extra.push(coord);
                let view = tmp.view(&w.occ);
                let ori_vec = match w.ori[&node] {
                    Axis::I => Cell::new(1, 0, 0),
                    Axis::J => Cell::new(0, 1, 0),
                    Axis::K => Cell::new(0, 0, 1),
                };
                let last_vec = path[1].vector_to(path[0]);
                let mut found = None;
                for sign in [1, -1] {
                    let od = ori_vec.cross(last_vec);
                    let od = Cell::new(sign * od.x, sign * od.y, sign * od.z);
                    let p0 = coord.add(od);
                    let p1 = p0.add(last_vec);
                    if !view.contains(&p0) && !view.contains(&p1) && !self.floors.outside(p0) && !self.floors.outside(p1) {
                        found = Some((p0, p1));
                        break;
                    }
                }
                let (p0, p1) = found?;
                w.push_path(vec![p1, p0, coord]);
                w.occ.insert(p0);
                w.occ.insert(p1);
                self.route_partners_from_cube(&mut w, node, 0, input)?;
            }
            5 => {
                let (_, _, inp) = self.route_input_ports(&mut w, node, coord, &self.input_connect[&node], 0)?;
                input = inp;
                w.t_track.insert(node, TTrack { exit: coord, path: vec![], ori: Some(w.ori[&node]) });
                self.route_partners_from_cube(&mut w, node, 0, input)?;
            }
            _ => panic!("unknown node type {nt}"),
        }

        Some(EmbeddingState::new(
            w.pos, w.ori, w.typ, w.paths, w.paths_max_z, w.occ, self.z_floor, self.floors, w.track, w.idle_place, w.t_track,
            self.layer.clone(), self.input_connect.clone(), self.order.clone(), self.z_length, self.htable.clone(), self.order_idx + 1,
        ))
    }

    /// `reward`: finish a complete layer (lifts to the ceiling, T exits) and
    /// score it; None if it is not terminal or some route fails.
    pub fn reward(&self, length: usize) -> Option<RewardResult> {
        if !self.is_terminal() {
            return None;
        }
        let typ = &self.typ;
        let node_type = &self.layer.node_type;
        let z_max = self.occupied.iter().map(|c| c.z).max().unwrap();
        let ceiling_z = z_max + 1;
        let num_ports = self.layer.output_connect.len();
        let edge_dist = 2;
        let port_loc = auto_ports(num_ports, ceiling_z, edge_dist, Some(length));

        let (x_min, x_max, y_min, y_max) = if num_ports == 0 {
            (self.floors.x_min.unwrap(), self.floors.x_max.unwrap(), self.floors.y_min.unwrap(), self.floors.y_max.unwrap())
        } else {
            let xs = port_loc.values().map(|c| c.x);
            let ys = port_loc.values().map(|c| c.y);
            (xs.clone().min().unwrap() as f64, xs.max().unwrap() as f64, ys.clone().min().unwrap() as f64, ys.max().unwrap() as f64)
        };
        let half = edge_dist as f64 / 2.0;
        let ceiling_floors = Floors::all(x_min - half, x_max + half, y_min - half, y_max + half);

        let mut node_target_pairs: NodeMap<Cell> = NodeMap::new();
        let mut available_nodes: Vec<NodeId> = self.layer.output_connect.keys().copied().collect();
        let available_targets: Vec<(usize, Cell)> = port_loc.iter().map(|(k, v)| (*k, *v)).collect();
        let mut pre_process: Vec<Cell> = Vec::new();
        for node in available_nodes.clone() {
            if typ[&node] == 2 {
                let p = self.pos[&node];
                let t = Cell::new(p.x, p.y, ceiling_z);
                if available_targets.iter().any(|(_, c)| *c == t) {
                    node_target_pairs.insert(node, t);
                    available_nodes.retain(|n| *n != node);
                    pre_process.push(t);
                }
            }
        }
        let mut sorted_port_indices: Vec<usize> = available_targets.iter().map(|(k, _)| *k).collect();
        sorted_port_indices.sort_unstable_by(|a, b| b.cmp(a));
        let target_of = |idx: usize| available_targets.iter().find(|(k, _)| *k == idx).unwrap().1;
        for &port_idx in &sorted_port_indices {
            if available_nodes.is_empty() {
                break;
            }
            let target = target_of(port_idx);
            if pre_process.contains(&target) {
                continue;
            }
            // min by (manhattan in x/y, -y); first wins on ties
            let mut best: Option<(i64, i64, NodeId)> = None;
            for &n in &available_nodes {
                let p = self.pos[&n];
                let key = (((p.x - target.x).abs() + (p.y - target.y).abs()) as i64, -(p.y as i64));
                if best.map_or(true, |(d, ny, _)| (key.0, key.1) < (d, ny)) {
                    best = Some((key.0, key.1, n));
                }
            }
            let closest = best.unwrap().2;
            node_target_pairs.insert(closest, target);
            available_nodes.retain(|n| *n != closest);
        }
        let mut node_order: Vec<NodeId> = Vec::new();
        for &port_idx in &sorted_port_indices {
            let t = target_of(port_idx);
            if let Some((n, _)) = node_target_pairs.iter().find(|(_, c)| **c == t) {
                node_order.push(*n);
            }
        }

        let mut occ_pre: Occ = self.occupied.clone();
        let mut ceiling_track: NodeMap<CeilingEntry> = NodeMap::new();
        let mut occ_ceiling: Occ = occ_pre.clone();
        for node in node_order {
            let nt = node_type[&node];
            let p = self.pos[&node];
            let target = node_target_pairs[&node];
            // occ_tmp = occ_ceiling + offsets (cubes) + the other ceiling targets
            let mut extra: Vec<Cell> = Vec::new();
            if is_cube(nt) {
                push_offsets(&mut extra, p, self.ori[&node]);
            }
            for (_, tp) in &available_targets {
                if *tp != target {
                    extra.push(*tp);
                }
            }
            let path = route_to_ceiling(p, &occ_ceiling, &extra, target, self.z_floor, ceiling_z as f64, ceiling_floors)?;
            let entry = if is_cube(nt) {
                if nt == 4 || nt == 5 {
                    let (ct, ld) = edge_tracer(&path, self.ori[&node], 0);
                    if ori_map(ld, ct, 0) == Axis::K {
                        CeilingEntry { path: path.clone(), ori: Some(ori_map(ld, ct, 1)), typ: 1 }
                    } else {
                        CeilingEntry { path: path.clone(), ori: Some(ori_map(ld, ct, 0)), typ: nt }
                    }
                } else {
                    let (ct, ld) = edge_tracer(&path, self.ori[&node], nt as u8);
                    if ori_map(ld, ct, nt as u8) == Axis::K {
                        let t2: u8 = if nt != 1 { 1 } else { 0 };
                        CeilingEntry { path: path.clone(), ori: Some(ori_map(ld, ct, t2)), typ: t2 as NodeType }
                    } else {
                        CeilingEntry { path: path.clone(), ori: Some(ori_map(ld, ct, nt as u8)), typ: nt }
                    }
                }
            } else {
                CeilingEntry { path: path.clone(), ori: None, typ: nt }
            };
            ceiling_track.insert(node, entry);
            occ_ceiling.insert(target);
            occ_ceiling.extend(interior(&path).iter().copied());
        }

        let mut new_t_track = self.t_track.clone();
        for (node, tr) in self.t_track.iter() {
            if !node_type.contains_key(node) {
                continue;
            }
            if (tr.exit.z as f64) < self.z_floor {
                continue; // unchanged
            }
            let idle_cells: Vec<Cell> = self.idle_place.values().copied().collect();
            let res = route_single_t_to_boundary(tr.exit, &mut occ_pre, &occ_ceiling, self.z_floor, ceiling_z as f64, ceiling_floors, tr.ori, 3, &idle_cells);
            let new_exit = res.target?;
            let new_path = res.path.unwrap();
            let combined = if tr.path.is_empty() {
                new_path
            } else {
                let mut c = tr.path.clone();
                c.extend_from_slice(&new_path[1..]);
                c
            };
            new_t_track.insert(*node, TTrack { exit: new_exit, path: combined, ori: res.ori });
        }
        Some(RewardResult { reward: -self.vol, t_track: new_t_track, occupied: occ_pre, ceiling_track })
    }
}

/// Helper for callers building an occupancy set from cells.
pub fn occ_from(cells: impl IntoIterator<Item = Cell>) -> Occ {
    cells.into_iter().collect::<FxHashSet<Cell>>()
}
