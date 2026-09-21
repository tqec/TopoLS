import math
import time

# ---------------------------------------------------------------------------
# MCTS node in the search tree
# ---------------------------------------------------------------------------

class MCTSNode:
    __slots__ = ("state","parent","children",
                 "visits","value","untried","try_flag", "id")

    def __init__(self, state, parent=None, move_num=None, block_switch=False, ceiling_switch=False):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried = state.moves(num=move_num, block_switch=block_switch, ceiling_switch=ceiling_switch)

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
        reward = rollout(node.state, layer=layer, block_switch=block_switch, ceiling_switch=ceiling_switch, length=length)
        if reward != -1e9:
            reward, rollout_state = reward
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

    # retrieve best completed embedding seen
    best_state = None
    best_val = -1e9
    stack = [root]
    while stack:
        n = stack.pop()
        if n.state.is_terminal():
            r = n.state.reward(length=length)
            if r is not None:
                r, _, _, _ = r
                if r > best_val:
                    best_val = r
                    best_state = n.state
        stack.extend(n.children)

    if best_state is None:
        best_state = best_rollout_state

    return best_state
