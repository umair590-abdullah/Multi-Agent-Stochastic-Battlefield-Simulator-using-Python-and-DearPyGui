import random
import copy
from collections import defaultdict
from dearpygui import dearpygui as dpg


######################stochastic outcomes per action##############
######################each tuple is equal to outcome name +probability#######

dieoutcomes = [
    ("failloss", 0.20),
    ("fail", 0.15),
    ("partial", 0.16),
    ("partialmove", 0.12),
    ("success", 0.26),
    ("critical", 0.11)
]

IMPASSABLE = {'X', '1', '2'}

# =========================
# STATE
# =========================
class Gamestate:
    def __init__(self, grid, rows, cols, rounds, energy, score, units):
        self.grid = grid
        self.rows = rows
        self.cols = cols
        self.rounds = rounds
        self.energy = energy
        self.score = score
        self.units = units
        self.eliminated = set()

    def clone(self):
        return copy.deepcopy(self)

# =========================
# AGENT
# =========================
class Agent:
    def __init__(self, name, depth):
        self.name = name
        self.depth = depth
        self.nodes = 0
        self.pruned = 0
        self.pruned_branches = defaultdict(int)

# =========================
# LOAD BOARD
# =========================
def loadboard(path):
    with open(path) as f:
        n, m, r = map(int, f.readline().split())
        # NOTE: if your board file separates cells with spaces (e.g. "X 1 2 . .")
        # use f.readline().split() instead of list(f.readline().strip())
        grid = [list(f.readline().strip()) for _ in range(n)]

        ax, ay = map(int, f.readline().split())
        bx, by = map(int, f.readline().split())
        cx, cy = map(int, f.readline().split())

    units = {
        "A": [(ay, ax)],
        "B": [(by, bx)],
        "C": [(cy, cx)]
    }

    return Gamestate(
        grid, n, m, r,
        {"A": 20, "B": 20, "C": 20},
        {"A": 0, "B": 0, "C": 0},
        units
    )

# =========================
# HELPERS
# =========================
def neighbors(x, y):
    return [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]

def inbounds(s, x, y):
    return 0 <= x < s.rows and 0 <= y < s.cols

def next_alive_turn(state, turn):
    """FIX: skip eliminated agents when rotating turns."""
    order = ("A", "B", "C")
    for i in range(1, 4):
        cand = (turn + i) % 3
        if order[cand] not in state.eliminated:
            return cand
    return turn  # nobody left alive (shouldn't happen, terminal() should catch it)

# =========================
# ACTIONS
# =========================
def getactions(state, agent):
    # FIX: eliminated agents can no longer act
    if agent.name in state.eliminated:
        return [("wait", None)]

    acts = []
    for (x,y) in state.units[agent.name]:
        for nx,ny in neighbors(x,y):
            if not inbounds(state,nx,ny): continue
            if state.grid[nx][ny] in IMPASSABLE: continue
            acts.append(("move",(x,y,nx,ny)))
    return acts if acts else [("wait",None)]

# =========================
# SIMULATION (move only, no stochastic effects)
# =========================
def simulate(state, agent, action):
    s = state.clone()
    if action[0] == "move":
        x,y,nx,ny = action[1]
        if (x,y) in s.units[agent.name]:
            s.units[agent.name].remove((x,y))
            s.units[agent.name].append((nx,ny))
    return s

# =========================
# OUTCOME - apply a resolved dice result to a state
# =========================
def apply_outcome(state, agent, outcome):
    s = state.clone()

    if outcome == "success":
        s.score[agent.name] += 3
    elif outcome == "partial":
        s.score[agent.name] += 1
    elif outcome == "critical":
        s.score[agent.name] += 5
    elif outcome == "failloss":
        s.energy[agent.name] -= 2
    elif outcome == "fail":
        s.energy[agent.name] -= 1

    if s.energy[agent.name] <= 0:
        s.eliminated.add(agent.name)

    return s

def roll_outcome():
    """FIX: actually roll the die for the real game (used to be search-only)."""
    names = [o[0] for o in dieoutcomes]
    weights = [o[1] for o in dieoutcomes]
    return random.choices(names, weights=weights, k=1)[0]

# =========================
# EVALUATION
# =========================
def evaluate(state, agent):
    others = [a for a in ("A","B","C") if a != agent.name]
    return state.score[agent.name] - sum(state.score[o] for o in others)/2

# =========================
# TERMINAL
# =========================
def terminal(state):
    alive = [a for a in ("A","B","C") if a not in state.eliminated]

    if len(alive) == 1:
        return True, alive[0]

    if state.rounds <= 0:
        winner = max(state.score, key=state.score.get)
        return True, winner

    return False, None

# =========================
# CHANCE NODE
# =========================
def chance(state, depth, agent, turn, alpha, beta, agents):
    agent.nodes += 1

    total = 0
    remaining = 1.0

    for outcome, prob in dieoutcomes:
        new_state = apply_outcome(state, agent, outcome)
        val = minimax(new_state, depth, agent, turn, alpha, beta, agents)

        total += prob * val
        remaining -= prob

        ub = total + remaining * beta
        lb = total + remaining * alpha

        if ub < alpha:
            agent.pruned += 1
            agent.pruned_branches[("UB_PRUNE", outcome)] += 1
            return ub

        if lb > beta:
            agent.pruned += 1
            agent.pruned_branches[("LB_PRUNE", outcome)] += 1
            return lb

    return total

# =========================
# MINIMAX
# =========================
def minimax(state, depth, agent, turn, alpha, beta, agents):
    agent.nodes += 1

    done, _ = terminal(state)
    if depth == 0 or done:
        return evaluate(state, agent)

    current = agents[turn]
    actions = getactions(state, current)

    if current.name == agent.name:
        best = float("-inf")
        for a in actions:
            next_turn = next_alive_turn(state, turn)  # FIX: skip dead agents
            val = chance(simulate(state,current,a),
                         depth-1,agent,next_turn,alpha,beta,agents)

            best = max(best,val)
            alpha = max(alpha,val)

            if beta <= alpha:
                agent.pruned += 1
                agent.pruned_branches["MAX_PRUNE"] += 1
                break
        return best

    else:
        best = float("inf")
        for a in actions:
            next_turn = next_alive_turn(state, turn)  # FIX: skip dead agents
            val = chance(simulate(state,current,a),
                         depth-1,agent,next_turn,alpha,beta,agents)

            best = min(best,val)
            beta = min(beta,val)

            if beta <= alpha:
                agent.pruned += 1
                agent.pruned_branches["MIN_PRUNE"] += 1
                break
        return best

# =========================
# DRAW
# =========================
def draw(state):
    dpg.delete_item("grid", children_only=True)
    dpg.delete_item("status", children_only=True)

    for r in range(state.rows):
        row = ""
        for c in range(state.cols):
            cell = state.grid[r][c]
            for u in state.units:
                if u not in state.eliminated and (r,c) in state.units[u]:
                    cell = u
            row += cell + " "
        dpg.add_text(row, parent="grid")

    # FIX: show score/energy/status so the GUI actually reflects game state
    dpg.add_text(f"Round left: {state.rounds}", parent="status")
    for name in ("A", "B", "C"):
        status = "ELIMINATED" if name in state.eliminated else "alive"
        dpg.add_text(
            f"{name}: score={state.score[name]}  energy={state.energy[name]}  ({status})",
            parent="status"
        )

# =========================
# STEP
# =========================
def step():
    global state, agents, turn

    done, winner = terminal(state)
    if done:
        print("\nGAME OVER. Winner:", winner)
        return

    agent = agents[turn]

    # FIX: skip a dead agent's turn entirely instead of letting it act
    if agent.name in state.eliminated:
        turn = next_alive_turn(state, turn)
        return

    agent.nodes = 0
    agent.pruned = 0
    agent.pruned_branches.clear()

    best = float("-inf")
    best_action = None

    for a in getactions(state, agent):
        next_turn = next_alive_turn(state, turn)
        val = chance(simulate(state,agent,a),
                     agent.depth-1,agent,next_turn,
                     float("-inf"),float("inf"),agents)

        if val > best:
            best = val
            best_action = a

    # apply the chosen move
    state = simulate(state, agent, best_action)

    # FIX: actually resolve a real dice roll for this turn and apply it
    # to the true game state (previously only ever happened inside the
    # hypothetical search tree, so score/energy never changed in real play)
    outcome = roll_outcome()
    state = apply_outcome(state, agent, outcome)

    state.rounds -= 1

    print(f"\nAgent {agent.name}  |  rolled: {outcome}")
    print("Nodes:", agent.nodes)
    print("Pruned:", agent.pruned)
    print("Pruned summary:")
    for k, v in agent.pruned_branches.items():
        print(f"{k} -> {v}")

    turn = next_alive_turn(state, turn)
    draw(state)

# =========================
# GUI
# =========================
def gui():
    dpg.create_context()

    with dpg.window(label="Battle"):
        dpg.add_button(label="Next", callback=step)
        dpg.add_group(tag="grid")
        dpg.add_separator()
        dpg.add_group(tag="status")

    draw(state)

    dpg.create_viewport(title="Game", width=600, height=600)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()

# =========================
# INIT
# =========================
# FIX: hardcoded absolute path replaced with a relative one so it isn't
# tied to one specific machine/user account. Put board.txt next to this script,
# or change this to wherever your board file actually lives.
path = "C:\\Users\\Umair abdullah\\Desktop\\notepad.txt"

state = loadboard(path)

agents = [
    Agent("A",4),
    Agent("B",3),
    Agent("C",2)
]

turn = 0

if __name__ == "__main__":
    gui()