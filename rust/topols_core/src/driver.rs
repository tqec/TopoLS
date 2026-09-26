//! `operation()` (driver.py): the layer loop, seed trials, block hand-off,
//! the fallback ladder (backtrack, ceiling retry, gate-by-gate, brute force)
//! and the final seal.
//!
//! The Python front end supplies every layer of the main graph and, for
//! every block, the re-layered graph the gate-by-gate fallback would build;
//! nothing here depends on pyzx.

use std::rc::Rc;
use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::embedding::fallback::{basic_embedding, BruteInput};
use crate::embedding::hadamard::HTable;
use crate::embedding::mcts::{mcts, SearchParams};
use crate::embedding::node::{ordered_edges, NodeId, NodeType};
use crate::embedding::ports::{auto_ports, ceiling, seal_brute_frontier, PORT_ORI};
use crate::embedding::state::{paths_max_z, CeilingEntry, EmbeddingState, Layer, NodeMap, Path, Paths, TTrack, Track};
use crate::geometry::{Cell, Floors};
use crate::pyrandom::PyRandom;
use crate::routing::astar::Occ;
use crate::routing::color::Axis;

/// `layer_info` output for one layer.
#[derive(Clone, Debug, Default)]
pub struct LayerData {
    pub input_connect: NodeMap<Vec<NodeId>>,
    pub inter_connect: Vec<(NodeId, NodeId)>,
    pub output_connect: NodeMap<i32>,
    pub node_type: NodeMap<NodeType>,
}

#[derive(Clone, Debug)]
pub struct IoEntry {
    pub kind: String,
    pub qubit: i64,
}

/// The re-layered graph of one block for the gate-by-gate fallback.
#[derive(Clone, Debug, Default)]
pub struct FallbackBlock {
    /// index j = block-local layer (0 = inherited frontier, unused)
    pub layers: Vec<LayerData>,
    pub qubit_of: FxHashMap<u32, i64>,
    pub io_info: Vec<(u32, IoEntry)>,
}

#[derive(Clone, Debug)]
pub struct Params {
    pub seed_init: u64,
    pub seed_step: usize,
    pub time_bound: f64,
    pub iter_num: usize,
    pub move_num: usize,
    pub length: usize,
    pub dir_opt: bool,
    pub backtrack: usize,
    pub z_floor: f64,
}

pub struct CompileInput {
    /// index i = layer; entry 0 unused
    pub layers: Vec<LayerData>,
    pub layer_to_block: Vec<u16>,
    pub q_num: usize,
    pub qubit_of: FxHashMap<u32, i64>,
    /// index = block
    pub blocks: Vec<FallbackBlock>,
    pub htable: HTable,
    pub io_info: Vec<(NodeId, IoEntry)>,
    pub params: Params,
}

pub struct CompileOutput {
    pub pos: NodeMap<Cell>,
    pub ori: NodeMap<Axis>,
    pub typ: NodeMap<NodeType>,
    pub paths: Vec<Path>,
    pub io_info: Vec<(NodeId, IoEntry)>,
    pub floors: Floors,
    pub volume: f64,
    pub x_length: f64,
    pub y_length: f64,
    pub z_length: f64,
}

// ---------------------------------------------------------------------------

/// `_Frontier`
#[derive(Clone)]
struct Frontier {
    pos: NodeMap<Cell>,
    ori: NodeMap<Axis>,
    typ: NodeMap<NodeType>,
    paths: Paths,
    paths_max_z: i32,
    occupied: Occ,
    z_floor: f64,
    idle_h_track: NodeMap<Track>,
    idle_place: NodeMap<Cell>,
    t_track: NodeMap<TTrack>,
}

impl Frontier {
    fn from_state(s: &EmbeddingState) -> Frontier {
        Frontier {
            pos: s.pos.clone(), ori: s.ori.clone(), typ: s.typ.clone(), paths: s.paths.clone(), paths_max_z: paths_max_z(s.paths.iter()), occupied: s.occupied.clone(),
            z_floor: s.z_floor, idle_h_track: s.idle_h_track.clone(), idle_place: s.idle_place.clone(), t_track: s.t_track.clone(),
        }
    }
    fn with_floor(mut self, z_floor: f64) -> Frontier {
        self.z_floor = z_floor;
        self
    }
    fn at_block_start(&self, input_connect: &NodeMap<Vec<NodeId>>, occupied_zmax: &Occ, z_floor: f64) -> Frontier {
        let mut keep: Vec<NodeId> = input_connect.values().flatten().copied().collect();
        keep.extend(self.idle_h_track.values().map(|t| t.origin));
        let pos: NodeMap<Cell> = self.pos.iter().filter(|(k, _)| keep.contains(k)).map(|(k, v)| (*k, *v)).collect();
        let mut occupied: Occ = pos.values().copied().collect();
        occupied.extend(occupied_zmax.iter().copied());
        Frontier {
            ori: self.ori.iter().filter(|(k, _)| keep.contains(k)).map(|(k, v)| (*k, *v)).collect(),
            typ: self.typ.iter().filter(|(k, _)| keep.contains(k)).map(|(k, v)| (*k, *v)).collect(),
            pos,
            paths: Paths::new(),
            paths_max_z: i32::MIN,
            occupied,
            z_floor,
            idle_h_track: self.idle_h_track.clone(),
            idle_place: self.idle_place.clone(),
            t_track: NodeMap::new(),
        }
    }
    fn as_state(&self, floors: Floors, ht: &Arc<HTable>) -> EmbeddingState {
        EmbeddingState::new(
            self.pos.clone(), self.ori.clone(), self.typ.clone(), self.paths.clone(), self.paths_max_z, self.occupied.clone(), self.z_floor, floors,
            self.idle_h_track.clone(), self.idle_place.clone(), self.t_track.clone(), Arc::new(Layer::default()), Arc::new(NodeMap::new()),
            Arc::new(vec![]), 1.0, ht.clone(), 0,
        )
    }
}

struct History {
    pos: NodeMap<Cell>,
    ori: NodeMap<Axis>,
    typ: NodeMap<NodeType>,
    paths: Vec<Path>,
}
impl History {
    fn record(&mut self, s: &EmbeddingState) {
        for (k, v) in &s.pos {
            self.pos.insert(*k, *v);
        }
        for (k, v) in &s.ori {
            self.ori.insert(*k, *v);
        }
        for (k, v) in &s.typ {
            self.typ.insert(*k, *v);
        }
        self.paths.extend(s.paths.iter().cloned());
    }
}

struct Cfg {
    seeds: Vec<u64>,
    iter_num: usize,
    time_bound: f64,
    move_nums: Vec<usize>,
    length: usize,
    floors: Floors,
    ht: Arc<HTable>,
}

/// `_search_layer`: one search rung. `input_connect` is the layer's shared
/// dict, shuffled in place by every seed exactly as in Python.
fn search_layer(
    cfg: &Cfg,
    front: &Frontier,
    input_connect: &mut NodeMap<Vec<NodeId>>,
    layer: &Arc<Layer>,
    z_length: f64,
    block_switch: bool,
    ceiling_switch: bool,
    idles_first: bool,
    base_keys: Option<&Vec<NodeId>>,
) -> (Option<EmbeddingState>, Vec<EmbeddingState>, Option<Vec<NodeId>>) {
    let mut best: Option<EmbeddingState> = None;
    let mut candidates: Vec<EmbeddingState> = Vec::new();
    let mut keys: Option<Vec<NodeId>> = base_keys.cloned();
    for &move_num in &cfg.move_nums {
        let mut jobs: Vec<(EmbeddingState, PyRandom)> = Vec::new();
        for &seed in &cfg.seeds {
            let mut rng = PyRandom::seed(seed);
            for (_, v) in input_connect.iter_mut() {
                rng.shuffle(v);
            }
            let input_connect_seed = Arc::new(input_connect.clone());
            if base_keys.is_none() {
                let mut k: Vec<NodeId> = layer.node_type.keys().copied().collect();
                rng.shuffle(&mut k);
                keys = Some(k);
            }
            let keys_now = keys.clone().unwrap();
            let mut order = keys_now.clone();
            let mut priority: Vec<NodeId> = Vec::new();
            if idles_first {
                for &k in &keys_now {
                    if layer.node_type[&k] != 2 {
                        continue;
                    }
                    let port = input_connect_seed[&k][0];
                    if matches!(front.typ[&port], 2 | 3) {
                        continue;
                    }
                    if front.ori[&port] != Axis::K {
                        priority.push(k);
                    }
                }
                let mut others: Vec<NodeId> = keys_now.iter().filter(|k| !priority.contains(k)).copied().collect();
                rng.shuffle(&mut others);
                order = priority.clone();
                order.extend(others);
            }
            let mut root = Some(EmbeddingState::new(
                front.pos.clone(), front.ori.clone(), front.typ.clone(), front.paths.clone(), front.paths_max_z, front.occupied.clone(), front.z_floor, cfg.floors,
                front.idle_h_track.clone(), front.idle_place.clone(), front.t_track.clone(), layer.clone(), input_connect_seed, Arc::new(order),
                z_length, cfg.ht.clone(), 0,
            ));
            for _ in 0..priority.len() {
                let r = root.take().unwrap();
                let mv = r.moves(&mut rng, 6, false, true, false);
                root = if mv.is_empty() { None } else { r.next_state(mv[0]) };
                if root.is_none() {
                    break;
                }
            }
            let Some(root) = root else { continue };
            jobs.push((root, rng));
        }
        let params = SearchParams { iters: cfg.iter_num, time_limit: Some(cfg.time_bound), move_num: Some(move_num), block_switch, ceiling_switch, length: cfg.length };
        // Every seed's search runs on its own thread (root parallelisation,
        // like the Python process pool); results are reduced in job order so
        // the first-wins tie-breaking is unchanged.
        let results: Vec<Option<EmbeddingState>> = std::thread::scope(|scope| {
            let handles: Vec<_> = jobs
                .into_iter()
                .map(|(root, mut rng)| {
                    let params = &params;
                    scope.spawn(move || mcts(Rc::new(root), &mut rng, params).map(|st| Rc::try_unwrap(st).unwrap_or_else(|rc| (*rc).clone())))
                })
                .collect();
            handles.into_iter().map(|h| h.join().expect("seed search panicked")).collect()
        });
        for st in results.into_iter().flatten() {
            if best.as_ref().map_or(true, |b| -st.vol > -b.vol) {
                best = Some(st.clone());
            }
            candidates.push(st);
        }
    }
    (best, candidates, keys)
}

/// `_commit_layer`: finish an embedded layer in place; the ceiling track.
fn commit_layer(state: &mut EmbeddingState, length: usize) -> Option<NodeMap<CeilingEntry>> {
    let r = state.reward(length)?;
    state.t_track = r.t_track;
    for tr in state.t_track.values() {
        state.paths.push_back(tr.path.clone());
    }
    state.occupied = r.occupied;
    Some(r.ceiling_track)
}

/// `_seal`
fn seal(
    hist: &mut History,
    brute_last: bool,
    pre_brute_state: Option<&EmbeddingState>,
    pre_state: &EmbeddingState,
    pre_ceiling_track: &NodeMap<CeilingEntry>,
    pre_node_type: &NodeMap<NodeType>,
    io_info: &mut Vec<(NodeId, IoEntry)>,
    io_extra: Option<&Vec<(NodeId, IoEntry)>>,
    floors: Floors,
) -> CompileOutput {
    let mut best = if brute_last {
        let mut s = pre_brute_state.unwrap().clone();
        seal_brute_frontier(&mut s);
        s
    } else {
        let mut s = pre_state.clone();
        let mut ct = pre_ceiling_track.clone();
        ceiling(&mut s, &mut ct, pre_node_type, true);
        s
    };
    let extra: Vec<Path> = best.idle_h_track.values().map(|t| t.path.clone()).collect();
    best.paths.extend(extra);
    hist.record(&best);
    if let Some(ex) = io_extra {
        for (k, v) in ex {
            if best.pos.contains_key(k) {
                if let Some(slot) = io_info.iter_mut().find(|(kk, _)| kk == k) {
                    slot.1 = v.clone();
                } else {
                    io_info.push((*k, v.clone()));
                }
            }
        }
    }
    let (x_length, y_length, z_length, volume) = crate::embedding::ports::calculate_space_time(&hist.pos, &hist.paths, &floors);
    CompileOutput {
        pos: std::mem::take(&mut hist.pos), ori: std::mem::take(&mut hist.ori), typ: std::mem::take(&mut hist.typ), paths: std::mem::take(&mut hist.paths),
        io_info: io_info.clone(), floors, volume, x_length, y_length, z_length,
    }
}

fn qubit_output_map(input_connect: &NodeMap<Vec<NodeId>>, qubit_of: &FxHashMap<u32, i64>) -> FxHashMap<i64, NodeId> {
    let mut m = FxHashMap::default();
    for key in input_connect.keys() {
        m.insert(qubit_of[&key.base], *key);
    }
    m
}

fn rename_layer(l: &LayerData, block: u16, rename_inputs: bool) -> LayerData {
    let r = |n: &NodeId| NodeId::in_block(n.base, block);
    LayerData {
        input_connect: l.input_connect.iter().map(|(k, vals)| (r(k), if rename_inputs { vals.iter().map(r).collect() } else { vals.clone() })).collect(),
        inter_connect: ordered_edges(l.inter_connect.iter().map(|(a, b)| (r(a), r(b)))),
        output_connect: l.output_connect.iter().map(|(k, v)| (r(k), *v)).collect(),
        node_type: l.node_type.iter().map(|(k, v)| (r(k), *v)).collect(),
    }
}

fn zmax_cells(occ: &Occ) -> (i32, Occ) {
    let m = occ.iter().map(|c| c.z).max().unwrap();
    (m, occ.iter().filter(|c| c.z == m).copied().collect())
}

// ---------------------------------------------------------------------------

/// `operation()`
pub fn operation(input: &CompileInput) -> CompileOutput {
    let p = &input.params;
    let ht = Arc::new(input.htable.clone());
    let mut io_info = input.io_info.clone();

    let edge_dist = 2;
    let ports = auto_ports(input.q_num, 0, edge_dist, Some(p.length));
    let xs = ports.values().map(|c| c.x);
    let ys = ports.values().map(|c| c.y);
    let half = edge_dist as f64 / 2.0;
    let floors = Floors::all(
        xs.clone().min().unwrap() as f64 - half, xs.max().unwrap() as f64 + half,
        ys.clone().min().unwrap() as f64 - half, ys.max().unwrap() as f64 + half,
    );
    let cfg = Cfg {
        seeds: (p.seed_init..p.seed_init + p.seed_step as u64).collect(),
        iter_num: p.iter_num, time_bound: p.time_bound,
        move_nums: if p.dir_opt { vec![1, p.move_num] } else { vec![1] },
        length: p.length, floors, ht: ht.clone(),
    };

    let mut front = Frontier {
        pos: ports.iter().map(|(i, c)| (NodeId::int(*i as u32), *c)).collect(),
        ori: ports.keys().map(|i| (NodeId::int(*i as u32), PORT_ORI)).collect(),
        typ: ports.keys().map(|i| (NodeId::int(*i as u32), 0)).collect(),
        paths: Paths::new(),
        paths_max_z: i32::MIN,
        occupied: ports.values().copied().collect(),
        z_floor: p.z_floor,
        idle_h_track: NodeMap::new(),
        idle_place: NodeMap::new(),
        t_track: NodeMap::new(),
    };

    let mut hist = History { pos: NodeMap::new(), ori: NodeMap::new(), typ: NodeMap::new(), paths: vec![] };
    let mut z_length = 1.0f64;
    let mut block: u16 = 0;
    let mut block_flag = false;
    let mut ceiling_flag = false;
    let mut backup_flag = false;
    let mut input_mapping_flag = false;
    let mut brute_to_block = false;
    let mut qubit_output_map_: FxHashMap<i64, NodeId> = FxHashMap::default();

    let mut layer_candidates: Vec<EmbeddingState>;
    let mut prev_candidates: Vec<EmbeddingState> = Vec::new();
    let mut prev_candidates_layer: i64 = -1;

    let mut brute_last = false;
    let mut pre_brute_state: Option<EmbeddingState> = None;

    let mut block_state = front.as_state(floors, &ht);
    let mut qubit_map_pre_layer: FxHashMap<i64, NodeId> = (0..input.q_num as i64).map(|q| (q, NodeId::int(q as u32))).collect();
    let mut occupied_zmax: Occ = Occ::default();

    let mut pre_state = front.as_state(floors, &ht);
    let mut pre_ceiling_track: NodeMap<CeilingEntry> = NodeMap::new();
    let mut pre_node_type: NodeMap<NodeType> = NodeMap::new();

    let mut best_state: Option<EmbeddingState> = None;
    let n_layers = input.layers.len();
    let last_block = *input.layer_to_block.iter().max().unwrap();

    for i in 1..n_layers {
        if backup_flag && input.layer_to_block[i] == block {
            continue;
        } else if backup_flag && input.layer_to_block[i] != block {
            backup_flag = false;
            input_mapping_flag = true;
        }

        let mut block_switch = i == 1;
        if input.layer_to_block[i] != block {
            block = input.layer_to_block[i];
            block_flag = true;
            ceiling_flag = true;
            block_switch = true;
        }

        let ld = &input.layers[i];
        let mut node_input_connect: NodeMap<Vec<NodeId>> = ld.input_connect.clone();
        if input_mapping_flag {
            node_input_connect = node_input_connect.keys().map(|k| (*k, vec![qubit_output_map_[&input.qubit_of[&k.base]]])).collect();
            input_mapping_flag = false;
        }
        let node_output_connect: NodeMap<i32> = ld.output_connect.iter().filter(|(_, v)| **v != 0).map(|(k, v)| (*k, *v)).collect();
        let node_type = ld.node_type.clone();

        if node_output_connect.is_empty() {
            return seal(&mut hist, brute_last, pre_brute_state.as_ref(), &pre_state, &pre_ceiling_track, &pre_node_type, &mut io_info, None, floors);
        }

        if block_flag {
            if !brute_to_block {
                let mut s = pre_state.clone();
                let mut ct = pre_ceiling_track.clone();
                ceiling(&mut s, &mut ct, &pre_node_type, false);
                best_state = Some(s);
            }
            brute_to_block = false;
        }

        if i > 1 {
            let bs = best_state.as_ref().unwrap();
            front = Frontier::from_state(bs);
            if block_flag {
                block_state = bs.clone();
                hist.record(bs);
                qubit_map_pre_layer = node_input_connect.iter().map(|(k, v)| (input.qubit_of[&k.base], v[0])).collect();
                let (block_max_z, zm) = zmax_cells(&bs.occupied);
                z_length = block_max_z as f64;
                occupied_zmax = zm;
                front = front.at_block_start(&node_input_connect, &occupied_zmax, block_max_z as f64);
                block_flag = false;
            }
        }

        let layer = Arc::new(Layer { node_type: node_type.clone(), inter_connect: ld.inter_connect.clone(), output_connect: node_output_connect.clone() });

        // Rung 0
        let (b, cands, last_keys) = search_layer(&cfg, &front, &mut node_input_connect, &layer, z_length, block_switch, block_switch, block_switch, None);
        best_state = b;
        layer_candidates = cands;
        if best_state.is_some() {
            ceiling_flag = false;
        }

        // Rung 1: backtrack
        if best_state.is_none() && p.backtrack >= 1 && !block_switch && !prev_candidates.is_empty() && prev_candidates_layer == i as i64 - 1 {
            let mut alternatives: Vec<(EmbeddingState, NodeMap<CeilingEntry>)> = Vec::new();
            for alt in prev_candidates.iter().take(p.backtrack) {
                let mut a = alt.clone();
                if let Some(ct) = commit_layer(&mut a, cfg.length) {
                    alternatives.push((a, ct));
                }
            }
            let mut found: Option<(EmbeddingState, NodeMap<CeilingEntry>, EmbeddingState, Vec<EmbeddingState>)> = None;
            'tiers: for ceiling_mode in [false, true] {
                for (alt, alt_ct) in &alternatives {
                    let start = if ceiling_mode {
                        let mut s = alt.clone();
                        let mut ct = alt_ct.clone();
                        ceiling(&mut s, &mut ct, &pre_node_type, false);
                        s
                    } else {
                        alt.clone()
                    };
                    let (cand, cands, _) = search_layer(&cfg, &Frontier::from_state(&start), &mut node_input_connect, &layer, z_length, false, ceiling_mode, ceiling_mode, None);
                    if let Some(c) = cand {
                        found = Some((alt.clone(), alt_ct.clone(), c, cands));
                        break 'tiers;
                    }
                }
            }
            if let Some((alt, alt_ct, c, cands)) = found {
                best_state = Some(c);
                layer_candidates = cands;
                pre_state = alt;
                pre_ceiling_track = alt_ct;
                ceiling_flag = false;
            }
        }

        if best_state.is_none() {
            // Rung 2: ceiling retry
            if !ceiling_flag {
                let mut cs = pre_state.clone();
                let mut ct = pre_ceiling_track.clone();
                ceiling(&mut cs, &mut ct, &pre_node_type, false);
                let (b, _, _) = search_layer(&cfg, &Frontier::from_state(&cs), &mut node_input_connect, &layer, z_length, block_switch, true, true, last_keys.as_ref());
                best_state = b;
            }

            // Rung 3: gate-by-gate
            if best_state.is_none() {
                backup_flag = true;
                let fb = &input.blocks[block as usize];
                let io_info_: Vec<(NodeId, IoEntry)> = fb.io_info.iter().map(|(v, e)| (NodeId::in_block(*v, block), e.clone())).collect();
                let n_rows = fb.layers.len();

                if n_rows <= 1 {
                    best_state = Some(pre_state.clone());
                    if block == last_block {
                        return seal(&mut hist, brute_last, pre_brute_state.as_ref(), &pre_state, &pre_ceiling_track, &pre_node_type, &mut io_info, Some(&io_info_), floors);
                    }
                }

                let (block_max_z, _) = zmax_cells(&block_state.occupied);
                z_length = block_max_z as f64;
                let mut z_floor = block_max_z as f64;
                front = Frontier::from_state(&block_state);
                let mut finished_qubits: Vec<i64> = Vec::new();
                let mut ceiling_state: Option<EmbeddingState> = None;
                let mut fb_keys: Option<Vec<NodeId>> = None;

                for j in 1..n_rows {
                    let raw = &fb.layers[j];
                    let mut nic: NodeMap<Vec<NodeId>> = raw.input_connect.clone();
                    let mut ntype: NodeMap<NodeType> = raw.node_type.clone();
                    let mut noc: NodeMap<i32> = raw.output_connect.clone();

                    if noc.is_empty() {
                        return seal(&mut hist, brute_last, pre_brute_state.as_ref(), &pre_state, &pre_ceiling_track, &pre_node_type, &mut io_info, Some(&io_info_), floors);
                    }

                    if j == 1 {
                        let mut nic_new: NodeMap<Vec<NodeId>> = NodeMap::new();
                        for key in nic.keys().copied().collect::<Vec<_>>() {
                            let q = fb.qubit_of[&key.base];
                            if let Some(sub) = qubit_map_pre_layer.get(&q) {
                                nic_new.insert(key, vec![*sub]);
                            } else {
                                finished_qubits.push(q);
                                ntype.shift_remove(&key);
                                noc.shift_remove(&key);
                            }
                        }
                        nic = nic_new;
                        front = front.at_block_start(&nic, &occupied_zmax, z_floor);
                        ceiling_flag = true;
                    } else {
                        for key in nic.keys().copied().collect::<Vec<_>>() {
                            if finished_qubits.contains(&fb.qubit_of[&key.base]) {
                                nic.shift_remove(&key);
                                ntype.shift_remove(&key);
                                noc.shift_remove(&key);
                            }
                        }
                    }
                    if j == n_rows - 1 {
                        noc = noc.keys().map(|k| (*k, 1)).collect();
                    }
                    let noc: NodeMap<i32> = noc.iter().filter(|(_, v)| **v != 0).map(|(k, v)| (*k, *v)).collect();

                    let renamed = rename_layer(&LayerData { input_connect: nic, inter_connect: raw.inter_connect.clone(), output_connect: noc, node_type: ntype }, block, j != 1);
                    let mut fnic = renamed.input_connect.clone();
                    let flayer = Arc::new(Layer { node_type: renamed.node_type.clone(), inter_connect: renamed.inter_connect.clone(), output_connect: renamed.output_connect.clone() });
                    if j > 1 {
                        front = Frontier::from_state(best_state.as_ref().unwrap()).with_floor(z_floor);
                    }

                    // Rung 3a
                    let (b, _, k) = search_layer(&cfg, &front, &mut fnic, &flayer, z_length, block_switch, false, false, None);
                    best_state = b;
                    fb_keys = k;
                    if best_state.is_some() {
                        ceiling_flag = false;
                    }

                    if best_state.is_none() {
                        // Rung 3b
                        if !ceiling_flag {
                            let mut cs = pre_state.clone();
                            let mut ct = pre_ceiling_track.clone();
                            ceiling(&mut cs, &mut ct, &pre_node_type, false);
                            z_floor = cs.z_floor;
                            let (b, _, _) = search_layer(&cfg, &Frontier::from_state(&cs), &mut fnic, &flayer, z_length, block_switch, true, true, fb_keys.as_ref());
                            best_state = b;
                            ceiling_state = Some(cs);
                        }

                        // Rung 4: brute force
                        if best_state.is_none() {
                            let (base, bf) = if j == 1 {
                                (block_state.clone(), Frontier::from_state(&block_state).at_block_start(&fnic, &occupied_zmax, z_floor))
                            } else if !ceiling_flag {
                                let cs = ceiling_state.as_ref().unwrap();
                                (cs.clone(), Frontier::from_state(cs).with_floor(z_floor))
                            } else {
                                ceiling_flag = false;
                                let pb = pre_brute_state.as_ref().unwrap();
                                (pb.clone(), Frontier::from_state(pb).with_floor(z_floor))
                            };
                            let bf_paths: Vec<Path> = bf.paths.iter().cloned().collect();
                            let out = basic_embedding(&BruteInput {
                                pos: &bf.pos, ori: &bf.ori, typ: &bf.typ, paths: &bf_paths, occupied: &bf.occupied, z_floor, floors,
                                idle_h_track: &bf.idle_h_track, t_track: &bf.t_track, node_type: &flayer.node_type, input_connect: &fnic,
                                inter_connect: &flayer.inter_connect, htable: &ht,
                            });
                            let mut bs = base;
                            bs.pos = out.pos;
                            bs.ori = out.ori;
                            bs.typ = out.typ;
                            bs.paths = out.paths.into_iter().collect();
                            bs.occupied = out.occupied;
                            bs.idle_h_track = out.idle_h_track;
                            bs.idle_place = out.idle_place;
                            bs.t_track = out.t_track;
                            ceiling_flag = true;
                            pre_brute_state = Some(bs.clone());
                            best_state = Some(bs);
                            brute_last = true;
                            if j == n_rows - 1 {
                                qubit_output_map_ = qubit_output_map(&fnic, &fb.qubit_of);
                                brute_to_block = true;
                            }
                            continue;
                        }
                    }

                    let bs = best_state.as_mut().unwrap();
                    pre_ceiling_track = commit_layer(bs, cfg.length).expect("reward() failed on an embedded layer");
                    brute_last = false;
                    pre_state = bs.clone();
                    pre_node_type = flayer.node_type.clone();
                    if j == n_rows - 1 {
                        qubit_output_map_ = qubit_output_map(&fnic, &fb.qubit_of);
                        for (k, v) in &io_info_ {
                            if bs.pos.contains_key(k) {
                                if let Some(slot) = io_info.iter_mut().find(|(kk, _)| kk == k) {
                                    slot.1 = v.clone();
                                } else {
                                    io_info.push((*k, v.clone()));
                                }
                            }
                        }
                    }
                }
                continue;
            }
            ceiling_flag = false;
        }

        // the layer is embedded
        let bs = best_state.as_mut().unwrap();
        pre_ceiling_track = commit_layer(bs, cfg.length).expect("reward() failed on an embedded layer");
        pre_state = bs.clone();
        pre_node_type = node_type;
        brute_last = false;
        if p.backtrack >= 1 {
            // Python drops the chosen state by identity; it is the first
            // candidate with the best volume (first-wins reduction).
            let best_vol = bs.vol;
            let mut others = layer_candidates;
            if let Some(idx) = others.iter().position(|c| c.vol == best_vol) {
                others.remove(idx);
            }
            others.sort_by(|a, b| a.vol.partial_cmp(&b.vol).unwrap());
            prev_candidates = others;
            prev_candidates_layer = i as i64;
        }
    }

    seal(&mut hist, brute_last, pre_brute_state.as_ref(), &pre_state, &pre_ceiling_track, &pre_node_type, &mut io_info, None, floors)
}
