# Deep Reinforcement Learning for MLIR Optimization: A Tutorial

This document serves as a foundational guide to understanding how Deep Reinforcement Learning (DRL) is applied in this project to optimize MLIR compiler code. It covers the basics of RL, the specific PPO algorithm used, and how these concepts map directly to the codebase.

---

## 1. Fundamentals of Deep Reinforcement Learning

At its core, Reinforcement Learning is about an **Agent** learning to make decisions by interacting with an **Environment**. 

At each time step $t$:
1. The agent observes the current **state** $s_t$.
2. It chooses an **action** $a_t$ based on its strategy (policy).
3. The environment transitions to a new state $s_{t+1}$.
4. The agent receives a **reward** $r_t$ indicating how good that action was.

The goal of the agent is to maximize the cumulative reward over time, called the **Return** $R_t$:
$$ R_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ... = \sum_{k=0}^{\infty} \gamma^k r_{t+k} $$
Here, $\gamma \in [0, 1)$ is the discount factor, which values immediate rewards more than distant, uncertain ones.

This process is modeled as a **Markov Decision Process (MDP)**, where the next state depends *only* on the current state and action, not the entire history.

### Types of RL Agents
* **Value-Based Methods (e.g., DQN):** The network predicts the expected return from a given state (the Value). The policy is simply to pick the action with the highest predicted value.
* **Policy-Based Methods (e.g., REINFORCE):** The network directly outputs a probability distribution over the actions given a state.
* **Actor-Critic Methods (e.g., PPO - Used in this project):** Combines both. The **Actor** focuses on what to do (the policy), and the **Critic** focuses on evaluating the state (the value).

---

## 2. Proximal Policy Optimization (PPO)

PPO is an Actor-Critic algorithm designed to update policies safely, preventing massive, destructive updates that cause the agent to forget its progress.

### The Advantage ($A_t$)
We first calculate the Advantage, which is the actual return minus the Critic's predicted value.
* $A_t > 0$: The action was better than expected (increase probability).
* $A_t < 0$: The action was worse than expected (decrease probability).
*(This project uses Generalized Advantage Estimation or GAE).*

### The Probability Ratio ($r_t$)
PPO compares the *new* probability of taking an action against the *old* probability:
$$ r_t(\theta) = \frac{\pi_{new}(a_t | s_t)}{\pi_{old}(a_t | s_t)} $$

### The Clipped Objective Function
PPO prevents the ratio $r_t$ from moving outside a specified range (e.g., $[0.8, 1.2]$). This prevents the policy from taking wildly large update steps:
$$ L^{CLIP} = \min \big( r_t \cdot A_t , \text{clip}(r_t, 1-\epsilon, 1+\epsilon) \cdot A_t \big) $$

### Total Loss
The total PPO loss combines:
1. **Policy Loss:** The clipped objective (encouraging good actions safely).
2. **Value Loss:** Mean Squared Error (training the Critic to make better predictions).
3. **Entropy Bonus:** Encourages exploration to prevent the agent from getting stuck in local optima.

---

## 3. The MLIR RL Agent Implementation

How does this RL theory translate into optimizing MLIR code? 

### State & Observation
The agent optimizes the code **one operation at a time**.
* **State (`OperationState`):** Contains the MLIR code, the current operation, producer data, and transformation history.
* **Observation (`Observation`):** A tensor representation of the code, encoding features like operation type, loop bounds, memory access matrices, and action history.

### Action Space (`rl_autoschedular/actions/`)
The agent uses a **Hierarchical Action Space**:
1. `Tiling`: Break large loops into smaller blocks.
2. `TiledParallelization`: Tile and execute outer loops on multiple cores.
3. `TiledFusion`: Tile and merge with a producer operation to improve memory locality.
4. `Interchange`: Swap the order of nested loops.
5. `Vectorization`: Utilize SIMD instructions.
6. `NoTransformation`: Move to the next operation.

Action masks prevent the agent from taking invalid actions (e.g., vectorizing tiny loops).

### Reward (`env.py`)
Rewards are **delayed** until the end of the episode sequence.
* The transformed code is compiled and timed (`execution.py`).
* Reward = $\log_{10}\left(\frac{\text{Original Time}}{\text{New Time}}\right)$
* Failed compilations yield large negative penalties (-5.0 or -20.0).

### Model Architecture (`model.py`)
* **Encoder (`LSTMEmbedding`):** Processes operation/producer features and embeds them using an LSTM to capture data flow dependencies.
* **Actor (`PolicyModel`):** Outputs probabilities for the 6 transformations and their parameters (e.g., tile sizes).
* **Critic (`ValueModel`):** Predicts the log-speedup of the current state.

### Training Loop (`ppo.py` & `train.py`)
1. **`collect_trajectory`:** Uses Dask to run many sub-environments, compiling transformed MLIR loops in parallel and fetching rewards.
2. **`value_update`:** Updates the Critic to predict returns more accurately.
3. **`ppo_update`:** Safely updates the Actor policy network using clipped probability ratios.
