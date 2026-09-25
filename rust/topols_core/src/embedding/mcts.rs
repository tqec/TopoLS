//! Anytime MCTS over `EmbeddingState`s (mcts.py): UCT selection, greedy
//! rollouts, best-so-far result within a work budget.

use std::rc::Rc;
use std::time::Instant;

use crate::embedding::state::EmbeddingState;
use crate::pyrandom::PyRandom;
use crate::routing::astar::{work, WORK_PER_SECOND};

/// The wall clock is only a safety net (`mcts.SAFETY_FACTOR`).
pub const SAFETY_FACTOR: f64 = 10.0;
pub const FAIL: f64 = -1e9;

struct Node {
    state: Rc<EmbeddingState>,
    parent: Option<usize>,
    children: Vec<usize>,
    visits: u64,
    value: f64,
    untried: Vec<crate::geometry::Cell>,
    /// `cached_reward`: Some(Some(r)) known reward, Some(None) known failure
    /// (reward() returned None), None not cached.
    cached: Option<Option<f64>>,
}

pub struct SearchParams {
    pub iters: usize,
    /// work budget in seconds of reference-machine search; None = unlimited
    pub time_limit: Option<f64>,
    pub move_num: Option<usize>,
    pub block_switch: bool,
    pub ceiling_switch: bool,
    pub length: usize,
}

fn new_node(arena: &mut Vec<Node>, state: Rc<EmbeddingState>, parent: Option<usize>, rng: &mut PyRandom, p: &SearchParams) -> usize {
    let untried = state.moves(rng, p.move_num.unwrap_or(6), p.block_switch, p.ceiling_switch, false);
    arena.push(Node { state, parent, children: vec![], visits: 0, value: 0.0, untried, cached: None });
    arena.len() - 1
}

/// `uct_select_child` with c = 0.7; ties go to the later child (`>=`).
fn uct_select(arena: &[Node], n: usize) -> usize {
    let c = 0.7f64;
    let mut best = None;
    let mut best_ucb = -1e9f64;
    let ln_n = (arena[n].visits as f64).ln();
    for &ch in &arena[n].children {
        let child = &arena[ch];
        let ucb = child.value / child.visits as f64 + c * (ln_n / child.visits as f64).sqrt();
        if ucb >= best_ucb {
            best_ucb = ucb;
            best = Some(ch);
        }
    }
    best.unwrap()
}

pub enum Rollout {
    Fail,
    /// reward, terminal state, whether that state is the rollout's input itself
    Ok(f64, Rc<EmbeddingState>, bool),
}

/// `rollout`: greedy completion by smallest bounding-box volume.
pub fn rollout(state: Rc<EmbeddingState>, rng: &mut PyRandom, p: &SearchParams) -> Rollout {
    let mut cur = state;
    let mut same = true;
    for _ in 0..2000 {
        if cur.is_terminal() {
            return match cur.reward(p.length) {
                None => Rollout::Fail,
                Some(r) => Rollout::Ok(r.reward, cur, same),
            };
        }
        let moves = cur.moves(rng, 6, p.block_switch, p.ceiling_switch, true);
        if moves.is_empty() {
            return Rollout::Fail;
        }
        let mut best_move = None;
        let mut best_vol = 1e9f64;
        for &m in &moves {
            if let Some(nxt) = cur.next_state(m) {
                if nxt.vol < best_vol {
                    best_vol = nxt.vol;
                    best_move = Some(m);
                }
            }
        }
        let Some(bm) = best_move else { return Rollout::Fail };
        // Python recomputes the chosen successor (and so does its work count).
        match cur.next_state(bm) {
            None => return Rollout::Fail,
            Some(n) => cur = Rc::new(n),
        }
        same = false;
    }
    Rollout::Fail
}

/// `mcts`: best terminal state found within the budget, or None.
pub fn mcts(root_state: Rc<EmbeddingState>, rng: &mut PyRandom, p: &SearchParams) -> Option<Rc<EmbeddingState>> {
    let mut arena: Vec<Node> = Vec::new();
    let root = new_node(&mut arena, root_state, None, rng, p);
    let work_budget = p.time_limit.map_or(f64::INFINITY, |t| t * WORK_PER_SECOND as f64);
    let work_start = work();
    let started = Instant::now();
    let deadline = p.time_limit.map_or(f64::INFINITY, |t| SAFETY_FACTOR * t);

    let mut best_rollout = FAIL;
    let mut best_rollout_state: Option<Rc<EmbeddingState>> = None;

    for _ in 0..p.iters {
        if (work() - work_start) as f64 >= work_budget || started.elapsed().as_secs_f64() > deadline {
            break;
        }
        // 1. selection
        let mut n = root;
        while arena[n].untried.is_empty() && !arena[n].children.is_empty() {
            n = uct_select(&arena, n);
        }
        // 2. expansion
        if let Some(mv) = arena[n].untried.pop() {
            let Some(nxt) = arena[n].state.next_state(mv) else { continue };
            let child = new_node(&mut arena, Rc::new(nxt), Some(n), rng, p);
            arena[n].children.push(child);
            n = child;
        }
        // 3. simulation
        let reward;
        let mut rollout_state: Option<Rc<EmbeddingState>> = None;
        if let (Some(Some(r)), true) = (arena[n].cached, arena[n].state.is_terminal()) {
            reward = r;
            rollout_state = Some(arena[n].state.clone());
        } else {
            match rollout(arena[n].state.clone(), rng, p) {
                Rollout::Fail => reward = FAIL,
                Rollout::Ok(r, st, same) => {
                    reward = r;
                    if same {
                        arena[n].cached = Some(Some(r));
                    }
                    rollout_state = Some(st);
                }
            }
        }
        if reward != FAIL && reward > best_rollout {
            best_rollout = reward;
            best_rollout_state = rollout_state;
        }
        // 4. back-propagation
        let mut m = Some(n);
        while let Some(i) = m {
            arena[i].visits += 1;
            arena[i].value += reward;
            m = arena[i].parent;
        }
    }

    // best completed embedding in the tree (DFS, stack order as in Python)
    let mut best_state: Option<Rc<EmbeddingState>> = None;
    let mut best_val = FAIL;
    let mut stack = vec![root];
    while let Some(i) = stack.pop() {
        if arena[i].state.is_terminal() {
            let r_val = match arena[i].cached {
                Some(c) => c,
                None => arena[i].state.reward(p.length).map(|r| r.reward),
            };
            if let Some(v) = r_val {
                if v > best_val {
                    best_val = v;
                    best_state = Some(arena[i].state.clone());
                }
            }
        }
        stack.extend(arena[i].children.iter().copied());
    }
    if best_state.is_none() || (best_rollout_state.is_some() && best_rollout > best_val) {
        best_state = best_rollout_state;
    }
    best_state
}
