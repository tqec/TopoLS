//! Replay routing calls recorded from the Python implementation
//! (dev/docs_diag/record_routing_fixtures.py) and require identical results.

use std::fs;
use std::path::PathBuf;

use serde_json::Value;
use topols_core::geometry::{Cell, Floors};
use topols_core::routing::astar::{shortest_path, Occ};
use topols_core::routing::boundary::route_single_t_to_boundary;
use topols_core::routing::color::{color_switch, edge_tracer, Axis};

fn fixture_dir() -> PathBuf {
    std::env::var("FIXTURE_DIR").map(PathBuf::from).unwrap_or_else(|_| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../dev/fixtures")
    })
}

fn records(prefix: &str) -> Vec<Value> {
    let mut out = Vec::new();
    let dir = fixture_dir();
    let Ok(entries) = fs::read_dir(&dir) else { return out };
    for e in entries.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        if name.starts_with(prefix) && name.ends_with(".jsonl") {
            for line in fs::read_to_string(e.path()).unwrap().lines() {
                if !line.trim().is_empty() {
                    out.push(serde_json::from_str(line).unwrap());
                }
            }
        }
    }
    out
}

fn cell(v: &Value) -> Cell {
    let a = v.as_array().unwrap();
    Cell::new(a[0].as_f64().unwrap() as i32, a[1].as_f64().unwrap() as i32, a[2].as_f64().unwrap() as i32)
}
fn cells(v: &Value) -> Vec<Cell> {
    v.as_array().unwrap().iter().map(cell).collect()
}
fn occ(v: &Value) -> Occ {
    cells(v).into_iter().collect()
}
fn floors(v: &Value) -> Floors {
    let a = v.as_array().unwrap();
    Floors { x_min: a[0].as_f64(), x_max: a[1].as_f64(), y_min: a[2].as_f64(), y_max: a[3].as_f64() }
}
fn opt_cells(v: &Value) -> Option<Vec<Cell>> {
    if v.is_null() { None } else { Some(cells(v)) }
}

#[test]
fn shortest_path_matches_python() {
    let recs = records("shortest_path");
    if recs.is_empty() {
        eprintln!("no recorded fixtures in {:?} (dev/docs_diag/record_routing_fixtures.py), skipped", fixture_dir());
        return;
    }
    let mut mismatches = 0;
    for r in &recs {
        let idle = cells(&r["idle_cells"]);
        let got = shortest_path(
            cell(&r["src"]), cell(&r["dst"]), &occ(&r["occupied"]), r["z_floor"].as_f64().unwrap(),
            floors(&r["floors"]), &idle, r["ceiling_z"].as_f64(), None,
        );
        if got != opt_cells(&r["result"]) {
            mismatches += 1;
            if mismatches <= 3 {
                eprintln!("MISMATCH shortest_path {:?}->{:?}\n python={:?}\n rust  ={:?}", cell(&r["src"]), cell(&r["dst"]), opt_cells(&r["result"]), got);
            }
        }
    }
    assert_eq!(mismatches, 0, "{mismatches}/{} shortest_path calls differ", recs.len());
    eprintln!("shortest_path: {} calls identical", recs.len());
}

#[test]
fn edge_tracer_matches_python() {
    let recs = records("edge_tracer");
    if recs.is_empty() {
        eprintln!("no recorded fixtures, skipped");
        return;
    }
    for r in &recs {
        let (t, ax) = edge_tracer(&cells(&r["path"]), Axis::from_char(r["ori"].as_str().unwrap().chars().next().unwrap()), r["face"].as_u64().unwrap() as u8);
        let want = r["result"].as_array().unwrap();
        assert_eq!((t as u64, ax.as_char().to_string()), (want[0].as_u64().unwrap(), want[1].as_str().unwrap().to_string()), "edge_tracer {:?}", r["path"]);
    }
    eprintln!("edge_tracer: {} calls identical", recs.len());
}

#[test]
fn color_switch_matches_python() {
    let recs = records("color_switch");
    if recs.is_empty() {
        eprintln!("no recorded fixtures, skipped");
        return;
    }
    let mut mismatches = 0;
    for r in &recs {
        let got = color_switch(&cells(&r["path"]), &occ(&r["occupied"]), r["z_floor"].as_f64().unwrap(), floors(&r["floors"]));
        if got != opt_cells(&r["result"]) {
            mismatches += 1;
            if mismatches <= 3 {
                eprintln!("MISMATCH color_switch\n python={:?}\n rust  ={:?}", opt_cells(&r["result"]), got);
            }
        }
    }
    assert_eq!(mismatches, 0, "{mismatches}/{} color_switch calls differ", recs.len());
    eprintln!("color_switch: {} calls identical", recs.len());
}

#[test]
fn route_t_matches_python() {
    let recs = records("route_T");
    if recs.is_empty() {
        eprintln!("no recorded fixtures, skipped");
        return;
    }
    let mut mismatches = 0;
    for r in &recs {
        let mut o: Occ = occ(&r["occ"]);
        let ori = match &r["ori"] { Value::String(s) => Some(Axis::from_char(s.chars().next().unwrap())), _ => None };
        let res = route_single_t_to_boundary(
            cell(&r["exit"]), &mut o, &occ(&r["occ_ceiling"]), r["z_floor"].as_f64().unwrap(), r["ceiling_z"].as_f64().unwrap(),
            floors(&r["floors"]), ori, 3, &cells(&r["idle_cells"]),
        );
        let want = &r["result"];
        let want_target = if want["target"].is_null() { None } else { Some(cell(&want["target"])) };
        let want_ori = match &want["ori"] { Value::String(s) => Some(Axis::from_char(s.chars().next().unwrap())), _ => None };
        let occ_after: Occ = occ(&want["occ_after"]);
        if res.target != want_target || res.path != opt_cells(&want["path"]) || res.ori != want_ori || o != occ_after {
            mismatches += 1;
            if mismatches <= 3 {
                eprintln!("MISMATCH route_T exit={:?}\n python target={:?} path={:?}\n rust   target={:?} path={:?}", cell(&r["exit"]), want_target, opt_cells(&want["path"]), res.target, res.path);
            }
        }
    }
    assert_eq!(mismatches, 0, "{mismatches}/{} route_T calls differ", recs.len());
    eprintln!("route_T: {} calls identical", recs.len());
}
