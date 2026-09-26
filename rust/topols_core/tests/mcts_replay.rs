//! Replay mcts() calls recorded from Python (dev/docs_diag/record_mcts_fixtures.py):
//! same root state, same RNG state, same budget -> the Rust search must
//! return the same best state.

use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;

use indexmap::IndexMap;
use rustc_hash::FxHashMap;
use serde_json::Value;
use topols_core::embedding::hadamard::HTable;
use topols_core::embedding::mcts::{mcts, SearchParams};
use topols_core::embedding::node::NodeId;
use topols_core::embedding::state::{occ_from, paths_max_z, EmbeddingState, Layer, Paths, TTrack, Track};
use topols_core::geometry::{Cell, Floors};
use topols_core::pyrandom::PyRandom;
use topols_core::routing::astar::work;
use topols_core::routing::color::Axis;

fn fixture_dir() -> PathBuf {
    std::env::var("FIXTURE_DIR").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../dev/fixtures"))
}

/// Parse a Python node id from its JSON form (int, or a string like
/// "12", "12_3", "12_old", "12_3_old").
fn node(v: &Value) -> NodeId {
    match v {
        Value::Number(n) => NodeId::int(n.as_u64().unwrap() as u32),
        Value::String(s) => node_str(s, false),
        _ => panic!("bad node id {v}"),
    }
}
fn node_str(s: &str, force_str: bool) -> NodeId {
    let parts: Vec<&str> = s.split('_').collect();
    let old = *parts.last().unwrap() == "old";
    let core: Vec<&str> = if old { parts[..parts.len() - 1].to_vec() } else { parts.clone() };
    let base: u32 = core[0].parse().unwrap();
    let block = if core.len() > 1 { Some(core[1].parse::<u16>().unwrap()) } else { None };
    let _ = force_str;
    NodeId { base, block, old }
}
/// Dicts are serialised with str keys; `keys` lists carry the original
/// int/str type in order. Build the id list for a dict from that.
fn keys(rec: &Value, name: &str) -> Vec<NodeId> {
    rec["keys"][name].as_array().unwrap().iter().map(node).collect()
}
fn cell(v: &Value) -> Cell {
    let a = v.as_array().unwrap();
    Cell::new(a[0].as_i64().unwrap() as i32, a[1].as_i64().unwrap() as i32, a[2].as_i64().unwrap() as i32)
}
fn cells(v: &Value) -> Vec<Cell> {
    v.as_array().unwrap().iter().map(cell).collect()
}
fn axis(v: &Value) -> Option<Axis> {
    v.as_str().map(|s| Axis::from_char(s.chars().next().unwrap()))
}

fn state_from(rec: &Value, ht: Arc<HTable>) -> EmbeddingState {
    let r = &rec["root"];
    let mut pos = IndexMap::new();
    for k in keys(r, "pos") { pos.insert(k, cell(&r["pos"][k.python_str()])); }
    let mut ori = IndexMap::new();
    for k in keys(r, "ori") { ori.insert(k, axis(&r["ori"][k.python_str()]).unwrap()); }
    let mut typ = IndexMap::new();
    for k in keys(r, "typ") { typ.insert(k, r["typ"][k.python_str()].as_i64().unwrap() as i8); }
    let paths: Paths = r["paths"].as_array().unwrap().iter().map(cells).collect();
    let pmz = paths_max_z(paths.iter());
    let occupied = occ_from(cells(&r["occupied"]));
    let fl = r["floors"].as_array().unwrap();
    let floors = Floors { x_min: fl[0].as_f64(), x_max: fl[1].as_f64(), y_min: fl[2].as_f64(), y_max: fl[3].as_f64() };
    let mut idle_h_track = IndexMap::new();
    for k in keys(r, "idle_h_track") {
        let t = &r["idle_h_track"][k.python_str()];
        idle_h_track.insert(k, Track { origin: node(&t[0]), path: cells(&t[1]), h: t[2].as_i64().unwrap() as i32 });
    }
    let mut idle_place = IndexMap::new();
    for k in keys(r, "idle_place") { idle_place.insert(k, cell(&r["idle_place"][k.python_str()])); }
    let mut t_track = IndexMap::new();
    for k in keys(r, "t_track") {
        let t = &r["t_track"][k.python_str()];
        t_track.insert(k, TTrack { exit: cell(&t[0]), path: cells(&t[1]), ori: axis(&t[2]) });
    }
    let mut node_type = IndexMap::new();
    for k in keys(r, "node_type") { node_type.insert(k, r["node_type"][k.python_str()].as_i64().unwrap() as i8); }
    let inter_connect: Vec<(NodeId, NodeId)> = r["inter_connect"].as_array().unwrap().iter().map(|e| (node(&e[0]), node(&e[1]))).collect();
    let mut output_connect = IndexMap::new();
    for k in keys(r, "output_connect") { output_connect.insert(k, r["output_connect"][k.python_str()].as_i64().unwrap() as i32); }
    let mut input_connect = IndexMap::new();
    for k in keys(r, "input_connect") {
        input_connect.insert(k, r["input_connect"][k.python_str()].as_array().unwrap().iter().map(node).collect::<Vec<_>>());
    }
    let order: Vec<NodeId> = r["order"].as_array().unwrap().iter().map(node).collect();
    let layer = Arc::new(Layer { node_type, inter_connect, output_connect });
    EmbeddingState::new(pos, ori, typ, paths, pmz, occupied, r["z_floor"].as_f64().unwrap(), floors, idle_h_track, idle_place, t_track,
                        layer, Arc::new(input_connect), Arc::new(order), r["z_length"].as_f64().unwrap(), ht, r["order_idx"].as_u64().unwrap() as usize)
}

fn htable_from(v: &Value) -> HTable {
    let mut h = HTable::default();
    for (q, rows) in v["rows_by_qubit"].as_object().unwrap() {
        h.rows_by_qubit.insert(q.parse().unwrap(), rows.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect());
    }
    for pair in v["cross"].as_array().unwrap() {
        let a = pair[0].as_array().unwrap();
        let b = pair[1].as_array().unwrap();
        h.add_cross((a[0].as_i64().unwrap(), a[1].as_f64().unwrap()), (b[0].as_i64().unwrap(), b[1].as_f64().unwrap()));
    }
    let mut qrow = FxHashMap::default();
    for (k, qr) in v["qrow"].as_object().unwrap() {
        qrow.insert(node_str(k, true), (qr[0].as_i64().unwrap(), qr[1].as_f64().unwrap()));
    }
    h.qrow = qrow;
    h
}

#[test]
fn mcts_calls_match_python() {
    let dir = fixture_dir();
    let mut recs: Vec<Value> = Vec::new();
    let Ok(entries) = std::fs::read_dir(&dir) else {
        eprintln!("no fixture dir {dir:?}, skipped");
        return;
    };
    for e in entries.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        if name.starts_with("mcts_") && name.ends_with(".jsonl") {
            for line in std::fs::read_to_string(e.path()).unwrap().lines() {
                if !line.trim().is_empty() { recs.push(serde_json::from_str(line).unwrap()); }
            }
        }
    }
    if recs.is_empty() {
        eprintln!("no recorded mcts fixtures in {dir:?} (dev/docs_diag/record_mcts_fixtures.py), skipped");
        return;
    }
    let (mut ok, mut bad) = (0, 0);
    for (i, rec) in recs.iter().enumerate() {
        let ht = Arc::new(htable_from(&rec["htable"]));
        let root = Rc::new(state_from(rec, ht));
        let words: Vec<u32> = rec["rng"]["words"].as_array().unwrap().iter().map(|w| w.as_u64().unwrap() as u32).collect();
        let mut rng = PyRandom::from_state(&words, rec["rng"]["index"].as_u64().unwrap() as usize);
        let p = &rec["params"];
        let params = SearchParams {
            iters: p["iters"].as_u64().unwrap() as usize, time_limit: p["time_limit"].as_f64(),
            move_num: p["move_num"].as_u64().map(|m| m as usize), block_switch: p["block_switch"].as_bool().unwrap(),
            ceiling_switch: p["ceiling_switch"].as_bool().unwrap(), length: p["length"].as_u64().unwrap() as usize,
        };
        let w0 = work();
        let res = mcts(root, &mut rng, &params);
        let used = work() - w0;
        let want = &rec["result"];
        let same = match (&res, want.is_null()) {
            (None, true) => true,
            (Some(s), false) => {
                let vol_ok = s.vol == want["vol"].as_f64().unwrap();
                let pos_ok = s.pos.iter().all(|(k, c)| want["pos"].get(k.python_str()).map_or(false, |v| cell(v) == *c)) && s.pos.len() == want["pos"].as_object().unwrap().len();
                let paths_ok = s.paths.len() == want["paths"].as_array().unwrap().len() && s.paths.iter().zip(want["paths"].as_array().unwrap()).all(|(a, b)| *a == cells(b));
                if !(vol_ok && pos_ok && paths_ok) {
                    eprintln!("call {i}: vol rust={} py={} pos_ok={pos_ok} paths_ok={paths_ok}", s.vol, want["vol"]);
                }
                vol_ok && pos_ok && paths_ok
            }
            (None, false) => { eprintln!("call {i}: rust None, python vol {}", want["vol"]); false }
            (Some(s), true) => { eprintln!("call {i}: rust vol {}, python None", s.vol); false }
        };
        let want_work = rec["work_used"].as_u64().unwrap();
        if used != want_work {
            eprintln!("call {i}: work rust={used} py={want_work} (result {})", if same { "same" } else { "DIFFERENT" });
        }
        if same { ok += 1 } else { bad += 1 }
    }
    eprintln!("mcts replay: {ok} identical, {bad} different (of {})", recs.len());
    assert_eq!(bad, 0);
}
