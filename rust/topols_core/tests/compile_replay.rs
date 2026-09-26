//! Whole-compile replay: run `driver::operation` on payloads dumped from
//! Python (dev/docs_diag/dump_compile_payload.py) and require the same
//! volume, node positions/orientations/types and paths.

use std::path::PathBuf;
use serde_json::Value;
use topols_core::driver::operation;
use topols_core::payload::parse_input;

fn fixture_dir() -> PathBuf {
    std::env::var("FIXTURE_DIR").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../dev/fixtures"))
}

fn check(name: &str) {
    let path = fixture_dir().join(format!("compile_{name}.json"));
    let Ok(text) = std::fs::read_to_string(&path) else {
        eprintln!("no fixture {path:?}, skipped");
        return;
    };
    let v: Value = serde_json::from_str(&text).unwrap();
    let input = parse_input(&v["payload"].to_string());
    let t0 = std::time::Instant::now();
    let out = operation(&input);
    let secs = t0.elapsed().as_secs_f64();
    let want = &v["expected"];
    let want_vol = want["volume"].as_f64().unwrap();
    let mut problems = Vec::new();
    if out.volume != want_vol {
        problems.push(format!("volume rust={} python={}", out.volume, want_vol));
    }
    let want_pos: Vec<(String, Vec<i64>)> = want["pos"].as_array().unwrap().iter().map(|e| (e[0].as_str().unwrap().to_string(), e[1].as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect())).collect();
    let got_pos: Vec<(String, Vec<i64>)> = out.pos.iter().map(|(k, c)| (k.python_str(), vec![c.x as i64, c.y as i64, c.z as i64])).collect();
    if got_pos != want_pos {
        problems.push(format!("pos differ ({} vs {} nodes)", got_pos.len(), want_pos.len()));
        for (g, w) in got_pos.iter().zip(want_pos.iter()).filter(|(g, w)| g != w).take(3) {
            eprintln!("  pos rust={g:?} python={w:?}");
        }
    }
    let want_ori: Vec<(String, String)> = want["ori"].as_array().unwrap().iter().map(|e| (e[0].as_str().unwrap().to_string(), e[1].as_str().unwrap().to_string())).collect();
    let got_ori: Vec<(String, String)> = out.ori.iter().map(|(k, a)| (k.python_str(), a.as_char().to_string())).collect();
    if got_ori != want_ori {
        problems.push("ori differ".into());
    }
    let want_paths: Vec<Vec<Vec<i64>>> = want["paths"].as_array().unwrap().iter().map(|p| p.as_array().unwrap().iter().map(|c| c.as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect()).collect()).collect();
    let got_paths: Vec<Vec<Vec<i64>>> = out.paths.iter().map(|p| p.iter().map(|c| vec![c.x as i64, c.y as i64, c.z as i64]).collect()).collect();
    if got_paths != want_paths {
        problems.push(format!("paths differ ({} vs {})", got_paths.len(), want_paths.len()));
    }
    use std::sync::atomic::Ordering::Relaxed;
    use topols_core::driver::{TOTAL_ASTAR_CALLS, TOTAL_ASTAR_NANOS, TOTAL_MCTS_NANOS, TOTAL_NS_CALLS, TOTAL_NS_NANOS, TOTAL_RECON_NANOS, TOTAL_REWARD_NANOS, TOTAL_SETUP_NANOS, TOTAL_WORK};
    let (w, ac, an, mn) = (TOTAL_WORK.swap(0, Relaxed), TOTAL_ASTAR_CALLS.swap(0, Relaxed), TOTAL_ASTAR_NANOS.swap(0, Relaxed), TOTAL_MCTS_NANOS.swap(0, Relaxed));
    let (nsn, nsc, rn) = (TOTAL_NS_NANOS.swap(0, Relaxed), TOTAL_NS_CALLS.swap(0, Relaxed), TOTAL_REWARD_NANOS.swap(0, Relaxed));
    let (setup, recon) = (TOTAL_SETUP_NANOS.swap(0, Relaxed), TOTAL_RECON_NANOS.swap(0, Relaxed));
    eprintln!("    A* setup {:.1} thread-s ({:.2} us/call), reconstruct {:.1} thread-s", setup as f64 / 1e9, setup as f64 / 1e3 / ac.max(1) as f64, recon as f64 / 1e9);
    eprintln!("    next_state: {} calls, {:.1} thread-s ({:.0}% of mcts, {:.1} us/call incl. A*); reward: {:.1} thread-s ({:.0}%); A* inside next_state+reward {:.1} thread-s", nsc, nsn as f64 / 1e9, 100.0 * nsn as f64 / mn.max(1) as f64, nsn as f64 / 1e3 / nsc.max(1) as f64, rn as f64 / 1e9, 100.0 * rn as f64 / mn.max(1) as f64, an as f64 / 1e9);
    eprintln!("{name}: rust volume {} in {secs:.1}s (python {want_vol}) {} | work {:.1}M in {:.1} thread-s ({:.1}M units/s); A* {} calls, {:.1} thread-s ({:.0}% of mcts), {:.0} expansions/call",
        out.volume, if problems.is_empty() { "IDENTICAL" } else { "DIFFERENT" }, w as f64 / 1e6, mn as f64 / 1e9, w as f64 / (mn as f64 / 1e9) / 1e6,
        ac, an as f64 / 1e9, 100.0 * an as f64 / mn.max(1) as f64, w as f64 / ac.max(1) as f64);
    assert!(problems.is_empty(), "{name}: {}", problems.join("; "));
}

#[test]
fn ghz_16() { check("ghz_16"); }
#[test]
fn bv_16() { check("bv_16"); }
#[test]
fn dj_16() { check("dj_16"); }
#[test]
fn vqe_16() { check("vqe_16"); }
#[test]
fn qaoa_16() { check("qaoa_16"); }
