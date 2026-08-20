# IMPLEMENTATION OF ON-POLICY MONTE CARLO CONTROL USING GYMNASIUM FROZENLAKE-V1

## Aim

To implement **On-Policy Monte Carlo Control** using the Gymnasium `FrozenLake-v1` environment and learn an improved policy by estimating the action-value function from complete episodes.

---

## Problem Statement

The `FrozenLake-v1` environment contains frozen tiles, holes, a starting position, and a goal position. The agent must learn a policy that allows it to reach the goal while avoiding the holes.

The objectives of this experiment are:

1. To generate complete episodes using the Gymnasium environment.
2. To estimate the action-value function `Q(s,a)` using Monte Carlo returns.
3. To use an epsilon-greedy policy for exploration and exploitation.
4. To improve the policy based on the learned Q-values.
5. To display the final Q-table, estimated state-value function, learned policy, and learning curve.

---

## Software Requirements

```bash
pip install gymnasium numpy matplotlib
```

---

## Environment Description

**FrozenLake-v1** is a grid-based reinforcement learning environment.

* The environment consists of a **4 × 4 grid**.
* The agent starts from the starting tile.
* The agent must move across frozen tiles.
* Some tiles contain holes, which terminate the episode.
* The goal tile provides a reward of `1`.
* Reaching a hole provides a reward of `0`.
* Other movements normally provide a reward of `0`.
* `is_slippery=False` is used so that the agent moves deterministically in the selected direction.

### Action Mapping

| Action | Direction |
| ------ | --------- |
| 0      | Left      |
| 1      | Down      |
| 2      | Right     |
| 3      | Up        |

---

## Theory

### Monte Carlo Method

Monte Carlo methods learn from **complete episodes**. Instead of updating the value function after every individual step, the agent waits until the episode finishes and then calculates the return obtained from each state-action pair.

An episode can be represented as:

`S₀, A₀, R₁, S₁, A₁, R₂, ..., Sₜ`

The return from a particular time step is the total discounted reward received from that point until the end of the episode.

The action-value function `Q(s,a)` represents the expected return obtained by taking action `a` in state `s` and then following the current policy.

The incremental update is:

`Q(s,a) ← Q(s,a) + α [Gₜ - Q(s,a)]`

Where:

| Symbol   | Meaning                   |
| -------- | ------------------------- |
| `s`      | Current state             |
| `a`      | Action taken              |
| `Gₜ`     | Return from time step `t` |
| `Q(s,a)` | Estimated action-value    |
| `α`      | Learning rate             |
| `γ`      | Discount factor           |

---

## On-Policy Monte Carlo Control

On-policy Monte Carlo Control learns the value of the same policy that is being used to generate the episodes.

The agent follows an **epsilon-greedy policy**. Initially, the agent performs more exploration. As learning progresses, epsilon is gradually reduced, causing the agent to select the best-known actions more frequently.

After each complete episode:

1. The episode history is stored.
2. The return is calculated backwards.
3. The corresponding Q-values are updated.
4. Epsilon is decreased gradually.
5. The improved Q-table is used for future action selection.

---

## Epsilon-Greedy Policy

Epsilon-greedy action selection balances **exploration** and **exploitation**.

* With probability `ε`, the agent chooses a random action.
* With probability `1 - ε`, the agent chooses the action having the highest Q-value.

The greedy action is selected using:

`a = argmax Q(s,a)`

After sufficient training, the final policy is obtained by selecting the action with the maximum Q-value for each state.

`π(s) = argmax Q(s,a)`

---

## Algorithm

1. Start the FrozenLake-v1 environment.
2. Initialize the Q-table with zeros.
3. Set the learning rate, discount factor, epsilon, and number of episodes.
4. Reset the environment to obtain the initial state.
5. Select an action using the epsilon-greedy policy.
6. Execute the action in the environment.
7. Store the state, action, and reward.
8. Continue until the episode terminates.
9. Traverse the stored episode backwards.
10. Calculate the return for each state-action pair.
11. Update the corresponding Q-value.
12. Reduce epsilon gradually while maintaining a minimum value.
13. Repeat the process for the specified number of episodes.
14. Calculate the state-value function from the maximum Q-value of each state.
15. Generate the final learned policy using the maximum Q-value.
16. Display the Q-table, state-value function, learned policy, average reward, and learning curve.
17. Close the environment.

---

# Python Program

## Monte Carlo Control

```python
# ============================================================
# IMPLEMENTATION OF ON-POLICY MONTE CARLO CONTROL
# USING GYMNASIUM FROZENLAKE-V1
# ============================================================

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. Create Environment
# ------------------------------------------------------------

env = gym.make(
    "FrozenLake-v1",
    is_slippery=False
)

n_states = env.observation_space.n
n_actions = env.action_space.n

print("Number of States :", n_states)
print("Number of Actions:", n_actions)


# ------------------------------------------------------------
# 2. Initialize Q-table
# ------------------------------------------------------------

Q = np.zeros((n_states, n_actions))


# ------------------------------------------------------------
# 3. Parameters
# ------------------------------------------------------------

alpha = 0.1
gamma = 0.99

epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9995

num_episodes = 10000

# Store reward obtained in every episode
episode_rewards = []


# ------------------------------------------------------------
# 4. Epsilon-Greedy Action Selection
# ------------------------------------------------------------

def epsilon_greedy_action(state, epsilon):

    # Exploration
    if np.random.random() < epsilon:
        return env.action_space.sample()

    # Exploitation
    return np.argmax(Q[state])


# ------------------------------------------------------------
# 5. Monte Carlo Control
# ------------------------------------------------------------

for episode in range(num_episodes):

    # Reset environment
    state, info = env.reset()

    # Store:
    # (state, action, reward)
    episode_data = []

    total_reward = 0

    terminated = False
    truncated = False


    # --------------------------------------------------------
    # Generate Complete Episode
    # --------------------------------------------------------

    while not terminated and not truncated:

        # Select action using epsilon-greedy policy
        action = epsilon_greedy_action(
            state,
            epsilon
        )

        # Execute action
        next_state, reward, terminated, truncated, info = env.step(
            action
        )

        # Store experience
        episode_data.append(
            (state, action, reward)
        )

        # Accumulate reward
        total_reward += reward

        # Move to next state
        state = next_state


    # --------------------------------------------------------
    # Calculate Return and Update Q-table
    # --------------------------------------------------------

    G = 0

    # Traverse the episode backwards
    for state, action, reward in reversed(episode_data):

        # Calculate discounted return
        G = reward + gamma * G

        # Incremental Monte Carlo update
        Q[state, action] = (
            Q[state, action]
            + alpha * (G - Q[state, action])
        )


    # --------------------------------------------------------
    # Store Episode Reward
    # --------------------------------------------------------

    episode_rewards.append(total_reward)


    # --------------------------------------------------------
    # Decay Epsilon
    # --------------------------------------------------------

    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )


# ============================================================
# 6. Calculate State-Value Function
# ============================================================

state_values = np.max(Q, axis=1)


# ============================================================
# 7. Calculate Learned Policy
# ============================================================

optimal_policy = np.argmax(Q, axis=1)


# ============================================================
# 8. Display Results
# ============================================================

def print_policy(policy):

    action_symbols = {
        0: "L",
        1: "D",
        2: "R",
        3: "U"
    }

    policy_grid = np.array(
        [
            action_symbols[action]
            for action in policy
        ]
    ).reshape(4, 4)

    print("\nName: Prakash C ")
    print("Register Number: 212223240122 ")

    print("\nLearned Policy:")
    print(policy_grid)


def print_value_function(values):

    print("\nEstimated State-Value Function:")

    print(
        np.round(
            values.reshape(4, 4),
            3
        )
    )


# ------------------------------------------------------------
# Final Q-table
# ------------------------------------------------------------

print("\nFinal Q-table:")

print(
    np.round(
        Q,
        3
    )
)


# ------------------------------------------------------------
# State-Value Function
# ------------------------------------------------------------

print_value_function(
    state_values
)


# ------------------------------------------------------------
# Learned Policy
# ------------------------------------------------------------

print_policy(
    optimal_policy
)


# ------------------------------------------------------------
# Average Reward
# ------------------------------------------------------------

average_reward = np.mean(
    episode_rewards[-1000:]
)

print(
    "\nAverage reward over last 1000 episodes:",
    round(average_reward, 3)
)


# ============================================================
# 9. Learning Curve
# ============================================================

window = 100

moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)


plt.figure(figsize=(10, 5))

plt.plot(
    moving_average
)

plt.xlabel("Episode")
plt.ylabel("Average Reward")

plt.title(
    "On-Policy Monte Carlo Control Learning Curve"
)

plt.grid(True)

plt.show()


# ============================================================
# 10. Close Environment
# ============================================================

env.close()
```

---

# Output
<img width="278" height="63" alt="image" src="https://github.com/user-attachments/assets/a60677d0-a954-427e-a2b2-7219789a143f" />

### Final Q-table:
<img width="386" height="401" alt="image" src="https://github.com/user-attachments/assets/a6daf0a8-b788-4f8e-b41c-781aa88616e4" />

### Estimated State-Value Function:
<img width="382" height="135" alt="image" src="https://github.com/user-attachments/assets/3c8088a2-4241-45cf-9ad6-004b6b761f66" />

### Learned Policy:
<img width="378" height="211" alt="image" src="https://github.com/user-attachments/assets/79629376-d066-4dbe-9dfd-d05d99eccc04" />

### Learning Curve:
<img width="1159" height="647" alt="image" src="https://github.com/user-attachments/assets/6aa28efb-20c3-423b-85c1-a7c17f84ea0e" />



---

# Result
The On-Policy Monte Carlo Control algorithm was successfully implemented using the Gymnasium FrozenLake-v1 environment. The agent learned Q-values and an improved policy through complete episode returns and epsilon-greedy exploration.

# Inference

The learned Q-table shows the estimated value of each action in every state. The average reward of 0.958 over the last 1000 episodes indicates that the agent learned an effective policy for reaching the goal while avoiding the holes.
