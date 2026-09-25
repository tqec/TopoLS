//! JSON encoding of `CompileInput` / `CompileOutput` (the interface used by
//! the Python bridge `topols.engine` and by the whole-compile tests).

use rustc_hash::FxHashMap;
use serde_json::{json, Value};

use crate::driver::{CompileInput, CompileOutput, FallbackBlock, IoEntry, LayerData, Params};
use crate::embedding::hadamard::HTable;
use crate::embedding::node::NodeId;
use crate::embedding::state::NodeMap;
use crate::geometry::Cell;

/// A node id as Python spells it: an int, or "v", "v_b", "v_old", "v_b_old".
pub fn parse_node(v: &Value) -> NodeId {
    match v {
        Value::Number(n) => NodeId::int(n.as_u64().unwrap() as u32),
        Value::String(s) => parse_node_str(s),
        _ => panic!("bad node id {v}"),
    }
}
pub fn parse_node_str(s: &str) -> NodeId {
    let parts: Vec<&str> = s.split('_').collect();
    let old = *parts.last().unwrap() == "old";
    let core = if old { &parts[..parts.len() - 1] } else { &parts[..] };
    let base: u32 = core[0].parse().unwrap_or_else(|_| panic!("bad node id {s}"));
    let block = if core.len() > 1 { Some(core[1].parse::<u16>().unwrap()) } else { None };
    NodeId { base, block, old }
}

fn layer(v: &Value) -> LayerData {
    let pairs = |name: &str| v[name].as_array().unwrap();
    LayerData {
        input_connect: pairs("input_connect").iter().map(|e| (parse_node(&e[0]), e[1].as_array().unwrap().iter().map(parse_node).collect())).collect(),
        inter_connect: pairs("inter_connect").iter().map(|e| (parse_node(&e[0]), parse_node(&e[1]))).collect(),
        output_connect: pairs("output_connect").iter().map(|e| (parse_node(&e[0]), e[1].as_i64().unwrap() as i32)).collect(),
        node_type: pairs("node_type").iter().map(|e| (parse_node(&e[0]), e[1].as_i64().unwrap() as i8)).collect(),
    }
}

fn io_entry(v: &Value) -> IoEntry {
    IoEntry { kind: v["type"].as_str().unwrap().to_string(), qubit: v["qubit"].as_i64().unwrap() }
}

pub fn parse_input(text: &str) -> CompileInput {
    let v: Value = serde_json::from_str(text).expect("payload is not JSON");
    let layers: Vec<LayerData> = v["layers"].as_array().unwrap().iter().map(layer).collect();
    let layer_to_block: Vec<u16> = v["layer_to_block"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u16).collect();
    let qubit_of: FxHashMap<u32, i64> = v["qubit_of"].as_array().unwrap().iter().map(|e| (e[0].as_u64().unwrap() as u32, e[1].as_i64().unwrap())).collect();
    let blocks: Vec<FallbackBlock> = v["blocks"]
        .as_array()
        .unwrap()
        .iter()
        .map(|b| FallbackBlock {
            layers: b["layers"].as_array().unwrap().iter().map(layer).collect(),
            qubit_of: b["qubit_of"].as_array().unwrap().iter().map(|e| (e[0].as_u64().unwrap() as u32, e[1].as_i64().unwrap())).collect(),
            io_info: b["io_info"].as_array().unwrap().iter().map(|e| (e[0].as_u64().unwrap() as u32, io_entry(&e[1]))).collect(),
        })
        .collect();
    let mut htable = HTable::default();
    let h = &v["htable"];
    for (q, rows) in h["rows_by_qubit"].as_object().unwrap() {
        htable.rows_by_qubit.insert(q.parse().unwrap(), rows.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect());
    }
    for pair in h["cross"].as_array().unwrap() {
        let (a, b) = (pair[0].as_array().unwrap(), pair[1].as_array().unwrap());
        htable.add_cross((a[0].as_i64().unwrap(), a[1].as_f64().unwrap()), (b[0].as_i64().unwrap(), b[1].as_f64().unwrap()));
    }
    for e in h["qrow"].as_array().unwrap() {
        htable.qrow.insert(parse_node(&e[0]), (e[1].as_i64().unwrap(), e[2].as_f64().unwrap()));
    }
    let io_info: Vec<(NodeId, IoEntry)> = v["io_info"].as_array().unwrap().iter().map(|e| (parse_node(&e[0]), io_entry(&e[1]))).collect();
    let p = &v["params"];
    let params = Params {
        seed_init: p["seed_init"].as_u64().unwrap(),
        seed_step: p["seed_step"].as_u64().unwrap() as usize,
        time_bound: p["time_bound"].as_f64().unwrap(),
        iter_num: p["iter_num"].as_u64().unwrap() as usize,
        move_num: p["move_num"].as_u64().unwrap() as usize,
        length: p["length"].as_u64().unwrap() as usize,
        dir_opt: p["dir_opt"].as_bool().unwrap_or(p["dir_opt"].as_i64().unwrap_or(1) == 1),
        backtrack: p["backtrack"].as_u64().unwrap() as usize,
        z_floor: p["z_floor"].as_f64().unwrap(),
    };
    CompileInput { layers, layer_to_block, q_num: v["q_num"].as_u64().unwrap() as usize, qubit_of, blocks, htable, io_info, params }
}

fn cell_json(c: &Cell) -> Value {
    json!([c.x, c.y, c.z])
}

fn map_json<V, F: Fn(&V) -> Value>(m: &NodeMap<V>, f: F) -> Value {
    Value::Array(m.iter().map(|(k, v)| json!([k.python_str(), f(v)])).collect())
}

pub fn output_json(o: &CompileOutput) -> String {
    let fl = &o.floors;
    json!({
        "pos": map_json(&o.pos, cell_json),
        "ori": map_json(&o.ori, |a| json!(a.as_char().to_string())),
        "typ": map_json(&o.typ, |t| json!(t)),
        "paths": o.paths.iter().map(|p| Value::Array(p.iter().map(cell_json).collect())).collect::<Vec<_>>(),
        "io_info": o.io_info.iter().map(|(k, e)| json!([k.python_str(), {"type": e.kind, "qubit": e.qubit}])).collect::<Vec<_>>(),
        "floors": [fl.x_min, fl.x_max, fl.y_min, fl.y_max],
        "volume": o.volume, "x_length": o.x_length, "y_length": o.y_length, "z_length": o.z_length,
    })
    .to_string()
}
