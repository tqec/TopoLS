use std::path::PathBuf;
use serde_json::Value;
use topols_core::pyrandom::PyRandom;

#[test]
fn matches_cpython_random() {
    // sequences recorded from CPython (`random.seed`, `getrandbits`, `shuffle`)
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/pyrandom.json");
    let recs: Vec<Value> = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    for r in &recs {
        let seed = r["seed"].as_u64().unwrap();
        let mut rng = PyRandom::seed(seed);
        let bits: Vec<u64> = (0..5).map(|_| rng.getrandbits(32)).collect();
        let want: Vec<u64> = r["bits"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
        assert_eq!(bits, want, "getrandbits seed {seed}");
        for (key, n) in [("shuffle10", 10), ("shuffle7", 7), ("shuffle1", 1), ("shuffle2", 2)] {
            let mut a: Vec<u64> = (0..n).collect();
            rng.shuffle(&mut a);
            let want: Vec<u64> = r[key].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
            assert_eq!(a, want, "{key} seed {seed}");
        }
        let (mt, idx) = rng.state_words();
        let want_first: Vec<u64> = r["state_first5"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
        assert_eq!(mt[..5].iter().map(|&w| w as u64).collect::<Vec<_>>(), want_first, "state seed {seed}");
        assert_eq!(idx as u64, r["state_index"].as_u64().unwrap(), "index seed {seed}");
    }
    eprintln!("pyrandom: {} seeds identical", recs.len());
}
