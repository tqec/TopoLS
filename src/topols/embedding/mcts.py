import math
import time

# ---------------------------------------------------------------------------
# Opt-in diagnostic hook (Phase 2, Step 2a -- see
# /home/junyuzh/.claude/plans/snappy-growing-aurora.md and
# docs/REFACTOR_LOG.md). Off by default (STATS_SINK is None): zero overhead,
# zero behavior change for production runs. A diagnostic script sets
# `topols.embedding.mcts.STATS_SINK = some_list` before calling `mcts()` to
# record, per call, whether the loop was cut off by the wall-clock
# `time_limit` ("search-bound") or exhausted its `iters` budget
# ("iters-bound") -- this determines which benchmarks are safe to use as
# exact-equality gates for optimizations that touch the timed loop.
STATS_SINK = None

# Separate opt-in sink for the cache-hit-rate diagnostic below (kept apart
# from STATS_SINK so the two don't get mixed into one list with
# differently-shaped dicts -- profile_boundedness.py indexes STATS_SINK
# entries by `["search_bound"]` unconditionally).
CACHE_STATS_SINK = None

# Tier 1 in-loop reward-cache shortcut (Phase 2 -- see
# docs/REFACTOR_LOG.md's "Step 2c, Tier 1 item 4" entry). Unlike Tier 0
# below, this DOES skip work inside the timed loop (it avoids calling
# rollout()/reward() again on an already-terminal node Selection revisits),
# so it can change how many iterations complete before `time_limit` for
# search-bound calls. Kept as a module toggle (default on) so a decoupled
# A/B comparison can flip it off without needing two copies of the code --
# see docs/profile_inloop_cache.py.
ENABLE_INLOOP_REWARD_CACHE = True

# Tier 0 optimization (Phase 2 -- see docs/REFACTOR_LOG.md "Step 2c" entry):
# sentinel for "no cached reward yet" on a tree node. Must be a distinct
# object, not `None` -- `EmbeddingState.reward()` legitimately returns
# `None` for a terminal state whose ceiling/T-gate routing failed, so
# overloading `None` as "not cached" would silently disable caching for
# exactly the states that fail routing and get revisited.
_UNSET = object()

# ---------------------------------------------------------------------------
# MCTS node in the search tree
# ---------------------------------------------------------------------------

class MCTSNode:
    __slots__ = ("state","parent","children",
                 "visits","value","untried","try_flag", "id", "cached_reward")

    def __init__(self, state, parent=None, move_num=None, block_switch=False, ceiling_switch=False):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried = state.moves(num=move_num, block_switch=block_switch, ceiling_switch=ceiling_switch)
        self.cached_reward = _UNSET

    def uct_select_child(self, c=0.7):
        best = None
        best_ucb = -1e9
        for child in self.children:
            ucb = (child.value/child.visits +
                   c * math.sqrt(math.log(self.visits) / child.visits))
            if ucb >= best_ucb:
                best_ucb = ucb
                best = child
        return best


# ---------------------------------------------------------------------------
# Roll‑out policy   (greedy placement to nearest free voxel)
# ---------------------------------------------------------------------------

def rollout(state, max_steps=2000, obj=None, layer=None, block_switch=False, ceiling_switch=False, length=4):

    cur = state

    for j in range(max_steps):
        if cur.is_terminal():
            result = cur.reward(layer=layer, length=length)
            if result is None:
                return -1e9
            reward, _, _, _ = result
            return reward, cur

        moves = cur.moves(num=6, block_switch=block_switch, ceiling_switch=ceiling_switch, rollout=True)

        if not moves:
            return -1e9

        # choose the move giving smallest immediate bbox growth
        best_move = None
        best_vol = 1e9

        for m in moves:
            nxt = cur.next_state(m)
            if nxt is None:
                continue
            if nxt.vol < best_vol:
                best_vol = nxt.vol
                best_move = m

        if best_move is None:
            return -1e9

        cur = cur.next_state(best_move)
        if cur is None:
            return -1e9

    return -1e9  # safeguard


# ---------------------------------------------------------------------------
# Main MCTS loop
# ---------------------------------------------------------------------------

def mcts(root_state, iters=10000, time_limit=None, obj=None, move_num=None, block_switch=False, ceiling_switch=False, layer=None, length=None):
    root = MCTSNode(root_state, move_num=move_num, block_switch=block_switch, ceiling_switch=ceiling_switch)
    end_time = time.time() + (time_limit if time_limit else 1e9)

    time_sel = 0
    time_exp = 0
    time_sim = 0
    time_bac = 0
    rollout_suc = 0
    best_rollout = -1e9
    best_rollout_state = None

    for i in range(iters):
        if time.time() > end_time:
            if STATS_SINK is not None:
                STATS_SINK.append({"iters_completed": i, "iters_requested": iters, "search_bound": True, "layer": layer})
            break

        node = root

        # 1. Selection
        t0 = time.time()
        while not node.untried and node.children:
            node = node.uct_select_child()
        t1 = time.time()
        time_sel = time_sel + (t1-t0)

        # 2. Expansion
        t0 = time.time()
        if node.untried:
            move = node.untried.pop()
            nxt_state = node.state.next_state(move)
            if nxt_state is None:
                continue
            node = MCTSNode(nxt_state, parent=node, move_num=move_num, block_switch=block_switch, ceiling_switch=ceiling_switch)
            node.parent.children.append(node)
        t1 = time.time()
        time_exp = time_exp + (t1-t0)

        # 3. Simulation
        t0 = time.time()
        if (ENABLE_INLOOP_REWARD_CACHE
                and node.cached_reward is not _UNSET
                and node.state.is_terminal()):
            # Tier 1 (see module docstring above): Selection walked back
            # down to a terminal leaf whose reward we already computed on
            # this exact, unmutated state object -- reuse it instead of
            # paying for rollout()'s call into reward() again. Equivalent
            # to what `rollout(node.state, ...)` would return (it would
            # immediately hit `if cur.is_terminal(): return cur.reward(...)`
            # with `cur is node.state`, unchanged), just without redoing
            # the routing work.
            reward = node.cached_reward
            rollout_state = node.state
            if CACHE_STATS_SINK is not None:
                CACHE_STATS_SINK.append({"cache_hit": True, "layer": layer})
        else:
            reward = rollout(node.state, layer=layer, block_switch=block_switch, ceiling_switch=ceiling_switch, length=length)
            if reward != -1e9:
                reward, rollout_state = reward
                # Tier 0: cache for the post-loop retrieval below (and, if
                # enabled, for a future in-loop revisit) -- a plain
                # attribute write, doesn't skip anything this iteration.
                if node.state is rollout_state:
                    node.cached_reward = reward
            if CACHE_STATS_SINK is not None and node.state.is_terminal():
                CACHE_STATS_SINK.append({"cache_hit": False, "layer": layer})
        if reward != -1e9:
            if reward > best_rollout:
                best_rollout = reward
                best_rollout_state = rollout_state
        t1 = time.time()
        time_sim = time_sim + (t1-t0)
        if reward > -1e9:
            rollout_suc = rollout_suc + 1

        # 4. Back‑propagation
        t0 = time.time()
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent
        t1 = time.time()
        time_bac = time_bac + (t1-t0)
    else:
        # Loop completed without `break` -- iters-bound, not search-bound.
        if STATS_SINK is not None:
            STATS_SINK.append({"iters_completed": iters, "iters_requested": iters, "search_bound": False, "layer": layer})

    # retrieve best completed embedding seen
    best_state = None
    best_val = -1e9
    stack = [root]
    while stack:
        n = stack.pop()
        if n.state.is_terminal():
            # Tier 0: reuse the reward cached during the loop above instead
            # of recomputing it (reward() does real routing work -- this is
            # strictly post-loop, so it cannot affect how many iterations
            # ran).
            if n.cached_reward is not _UNSET:
                r_val = n.cached_reward
            else:
                r = n.state.reward(length=length)
                r_val = r[0] if r is not None else None
            if r_val is not None and r_val > best_val:
                best_val = r_val
                best_state = n.state
        stack.extend(n.children)

    # Return the best state seen ANYWHERE, not "a terminal node in the tree if
    # one exists, else the best rollout". The tree only contains terminals
    # that happened to be expanded, and the first one to appear is usually
    # poor; `best_rollout_state` is the best over every rollout so far. The
    # old preference for the tree terminal meant a LONGER search (deeper
    # tree, first terminal appears) could return a worse state than a shorter
    # one -- measured 2026-09-24: dj_16 648 at -t 2 but 1215 at -t 5 (job
    # 4817). With max() over both, the result is the best-so-far of a
    # deterministic iteration sequence, so more time / iterations / seeds can
    # never return a worse state for the same (seed, state).
    if best_state is None or (best_rollout_state is not None and best_rollout > best_val):
        best_state = best_rollout_state

    return best_state
