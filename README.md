**Multi-Agent Strategy Game AI (Expectiminimax with Alpha-Beta Pruning)**
**What it is?**
-A turn-based strategy game where three AI-controlled agents (A, B, and C) compete against each other on a grid map. 
-Each agent moves around the board and takes actions, but the outcome of every action is uncertain — much like rolling a die.
-An action might succeed brilliantly, partially succeed, fail, or even backfire and cost the agent energy. 
-The goal for each agent is to score as many points as possible while staying alive, since running out of energy eliminates them from the game.
**The core challenge I solved:**
Because outcomes are random, a "smart" AI can't just calculate one fixed path to victory — it has to plan for every possible outcome of every move, weigh them by probability,
and choose the action that gives the best expected result over time, all while accounting for two opponents also trying to win.
This is a classic decision-making-under-uncertainty problem.
**How it works, step by step:**

**1.Decision-making engine:**
I built an expectiminimax search algorithm — an extension of the minimax algorithm used in games like chess,
but adapted to handle randomness. Instead of just alternating between "my best move" and "opponent's best move,"
the algorithm also has to average over all possible random outcomes at "chance nodes" before deciding.

**2.Performance optimization:**
Searching every possible future outcome gets expensive very fast, so I implemented a custom form of alpha-beta pruning — a technique that lets the 
algorithm skip exploring branches that it can mathematically prove won't change the final decision. 
I extended this pruning to work even at the probabilistic "chance" steps, using running probability bounds to cut off unnecessary calculations early.
**3.Game rules & state tracking:** The game engine tracks each agent's position, score, remaining energy, and elimination status, and enforces rules like blocked terrain, turn order, and game-ending conditions (last agent standing, or highest score when the round limit is reached).



https://github.com/user-attachments/assets/dc0b0404-98fd-418e-9ec4-59c5f7db9627


**4.Visualization:** I built an interactive GUI (using DearPyGui) that displays the live grid, agent positions, 
and each agent's current score/energy/status, updating in real time as the game progresses turn by turn.

**Skills demonstrated:**

-Algorithm design (adversarial search, probabilistic decision trees, pruning/optimization techniques)
-Python object-oriented programming (state management, cloning/simulation of game states)
-Debugging and correctness — as this project evolved, I also went back and fixed a subtle logic bug where the
AI's hypothetical planning outcomes weren't being properly applied to the real game state, 
which is a good example of the kind of careful state-management bugs that come up in simulation and game-engine code.
-GUI/data visualization integration
