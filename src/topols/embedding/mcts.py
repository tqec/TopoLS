"""Anytime Monte Carlo Tree Search over `EmbeddingState`s: UCT selection,
greedy rollouts, and a best-so-far result within an iteration and
wall-clock budget.
"""

import math
import time

# ---------------------------------------------------------------------------
# Optional diagnostics: a caller may set STATS_SINK to a list to record, per
# mcts() call, how many iterations completed and whether the wall-clock budget
# or the iteration cap ended the search. None (the default) records nothing.
STATS_SINK = None

# Separate opt-in sink for the reward-cache hit-rate diagnostic below.
CACHE_STATS_SINK = None

# In-loop reward cache: when selection lands on a terminal node whose reward
# is already known, reuse it instead of rerunning rollout()/reward(). A module
# toggle so it can be switched off for comparisons.
ENABLE_INLOOP_REWARD_CACHE = True

# Sentinel for "no cached reward yet". Distinct from None because reward()
# legitimately returns None when a terminal state's ceiling or T-gate routing
# fails, and those states must stay cacheable.
_UNSET = object()

# ---------------------------------------------------------------------------
# MCTS node in the search tree
# ---------------------------------------------------------------------------

class MCTSNode:
    """A node of the search tree.

    Attributes:
        state: the `EmbeddingState` at this node.
        parent, children: tree links.
        visits, value: visit count and summed backed-up reward (UCT statistics).
        untried: moves of `state` not expanded yet (`EmbeddingState.moves`).
        cached_reward: reward of `state` once known (terminal states only);
            `_UNSET` until then.
    """
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
        """Child with the highest UCB1 score, `mean value + c * sqrt(ln N / n)`."""
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
    """Complete `state` greedily: at every step take the move whose
    resulting state has the smallest bounding-box volume.

    Returns:
        `(reward, terminal_state)` when the layer is completed and
        `EmbeddingState.reward` succeeds; the scalar `-1e9` otherwise
        (no legal move, a routing failure, or `max_steps` exhausted).
    """
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
    """Search for the lowest-volume complete embedding of one layer.

    Standard UCT loop (select, expand one child, greedy rollout, back up)
    over `EmbeddingState`s, stopped by whichever of `iters` iterations or
    `time_limit` seconds comes first. The search is *anytime*: for a fixed
    random state and `root_state` the iteration sequence is deterministic,
    and the result is the best complete state seen anywhere (in the tree
    or in a rollout), so a larger budget never returns a worse state.

    Args:
        root_state: `EmbeddingState` with the previous layer embedded and
            nothing of the current layer placed yet.
        iters: maximum number of iterations.
        time_limit: wall-clock budget in seconds (None = unlimited).
        move_num: number of candidate placements generated per node
            (`EmbeddingState.moves(num=...)`).
        block_switch, ceiling_switch: passed to `moves`; the first block
            of a compile and a search started from a lifted ceiling allow
            different placements.
        layer, length: forwarded to `EmbeddingState.reward`.

    Returns:
        The best terminal `EmbeddingState`, or None if no rollout completed
        the layer.
    """
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
            # Selection reached a terminal node whose reward is already cached;
            # rollout() would recompute exactly this value.
            reward = node.cached_reward
            rollout_state = node.state
            if CACHE_STATS_SINK is not None:
                CACHE_STATS_SINK.append({"cache_hit": True, "layer": layer})
        else:
            reward = rollout(node.state, layer=layer, block_switch=block_switch, ceiling_switch=ceiling_switch, length=length)
            if reward != -1e9:
                reward, rollout_state = reward
                # Cache the reward for the post-loop retrieval and later revisits.
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
            # Reuse the cached reward (reward() does real routing work).
            if n.cached_reward is not _UNSET:
                r_val = n.cached_reward
            else:
                r = n.state.reward(length=length)
                r_val = r[0] if r is not None else None
            if r_val is not None and r_val > best_val:
                best_val = r_val
                best_state = n.state
        stack.extend(n.children)

    # Return the best state seen anywhere. The tree only holds the terminals
    # that happened to be expanded (the first to appear is often poor), while
    # best_rollout_state is the best over every rollout; taking the better of
    # the two makes the result the best-so-far of a deterministic iteration
    # sequence, so a longer budget can never return a worse state.
    if best_state is None or (best_rollout_state is not None and best_rollout > best_val):
        best_state = best_rollout_state

    return best_state
