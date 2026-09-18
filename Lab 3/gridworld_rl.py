import random

# ============================================================
# GRIDWORLD ENVIRONMENT
# ============================================================

# Grid:
#
# S1 | S2 | S3
# S4 | S5 | G
#
# Coordinates:
# S1 = (0,0)
# S2 = (0,1)
# S3 = (0,2)
# S4 = (1,0)
# S5 = (1,1)
# G  = (1,2)

states = ["S1", "S2", "S3", "S4", "S5", "G"]

actions = ["U", "D", "L", "R"]

terminal_state = "G"

# State coordinates
coordinates = {
    "S1": (0, 0),
    "S2": (0, 1),
    "S3": (0, 2),
    "S4": (1, 0),
    "S5": (1, 1),
    "G":  (1, 2)
}

# Reverse mapping
state_from_coordinate = {
    value: key for key, value in coordinates.items()
}


# ============================================================
# ENVIRONMENT TRANSITION FUNCTION
# ============================================================

def step(state, action):

    # Goal is terminal
    if state == terminal_state:
        return state, 0

    row, col = coordinates[state]

    # Calculate new position
    if action == "U":
        new_row, new_col = row - 1, col

    elif action == "D":
        new_row, new_col = row + 1, col

    elif action == "L":
        new_row, new_col = row, col - 1

    elif action == "R":
        new_row, new_col = row, col + 1

    else:
        raise ValueError("Invalid action")

    # If agent tries to leave grid,
    # it stays in the same state
    if (new_row, new_col) not in state_from_coordinate:
        return state, -1

    next_state = state_from_coordinate[(new_row, new_col)]

    # Reaching goal gives +10
    if next_state == terminal_state:
        return next_state, 10

    # Normal movement gives -1
    return next_state, -1


# ============================================================
# TASK 2: POLICY EVALUATION
# ============================================================

# Given policy
#
# This policy generally moves the agent toward the goal.
#
# S1 -> Right
# S2 -> Right
# S3 -> Down
# S4 -> Right
# S5 -> Right

policy = {
    "S1": "R",
    "S2": "R",
    "S3": "D",
    "S4": "R",
    "S5": "R"
}


def policy_evaluation(policy, gamma=0.9, theta=0.001, max_iterations=5):

    V = {state: 0.0 for state in states}

    iteration_data = []

    for iteration in range(1, max_iterations + 1):

        delta = 0

        new_V = V.copy()

        for state in states:

            if state == terminal_state:
                continue

            action = policy[state]

            next_state, reward = step(state, action)

            new_value = reward + gamma * V[next_state]

            delta = max(delta, abs(new_value - V[state]))

            new_V[state] = new_value

        V = new_V

        convergence = "Converged" if delta < theta else "Not Converged"

        iteration_data.append(
            (iteration, delta, convergence)
        )

    return V, iteration_data


# Run policy evaluation

values, iteration_data = policy_evaluation(policy)


print("\n========================================")
print("TASK 2: POLICY EVALUATION")
print("========================================")

print("\nIteration    Maximum Change (Delta)    Status")

for iteration, delta, status in iteration_data:
    print(
        f"{iteration:<12} {delta:<25.4f} {status}"
    )


print("\nState Values under Evaluated Policy:")

for state in states:
    print(f"{state}: {values[state]:.4f}")


# ============================================================
# TASK 3: VALUE ITERATION
# ============================================================

def value_iteration(gamma=0.9, theta=0.001):

    V = {state: 0.0 for state in states}

    iteration = 0

    while True:

        iteration += 1

        delta = 0

        new_V = V.copy()

        for state in states:

            if state == terminal_state:
                continue

            action_values = []

            for action in actions:

                next_state, reward = step(state, action)

                value = reward + gamma * V[next_state]

                action_values.append(value)

            best_value = max(action_values)

            new_V[state] = best_value

            delta = max(
                delta,
                abs(best_value - V[state])
            )

        V = new_V

        if delta < theta:
            break

    # Extract optimal policy
    optimal_policy = {}

    for state in states:

        if state == terminal_state:
            continue

        best_action = None
        best_value = float("-inf")

        for action in actions:

            next_state, reward = step(state, action)

            value = reward + gamma * V[next_state]

            if value > best_value:

                best_value = value
                best_action = action

        optimal_policy[state] = best_action

    return V, optimal_policy, iteration


optimal_values, optimal_policy, vi_iterations = value_iteration()


print("\n========================================")
print("TASK 3: VALUE ITERATION")
print("========================================")

print(f"\nValue Iteration converged after {vi_iterations} iterations.")

print("\nState    Optimal Action    State Value")

for state in states:

    if state == terminal_state:
        continue

    print(
        f"{state:<9} {optimal_policy[state]:<17} "
        f"{optimal_values[state]:.4f}"
    )


# ============================================================
# DISPLAY POLICY IN HUMAN-READABLE FORM
# ============================================================

action_names = {
    "U": "Up",
    "D": "Down",
    "L": "Left",
    "R": "Right"
}

print("\nOptimal Policy:")

for state in states:

    if state != terminal_state:
        print(
            f"{state} -> {action_names[optimal_policy[state]]}"
        )


# ============================================================
# TASK 4: PATH GENERATION
# ============================================================

def generate_path(policy, start_state="S1", max_steps=20):

    current_state = start_state

    path = [current_state]

    total_reward = 0

    for step_number in range(max_steps):

        if current_state == terminal_state:
            break

        action = policy[current_state]

        next_state, reward = step(current_state, action)

        total_reward += reward

        current_state = next_state

        path.append(current_state)

        if current_state == terminal_state:
            break

    goal_reached = current_state == terminal_state

    return path, len(path) - 1, total_reward, goal_reached


# ============================================================
# RANDOM POLICY
# ============================================================

random_policy = {
    state: random.choice(actions)
    for state in states
    if state != terminal_state
}


# ============================================================
# EVALUATED POLICY
# ============================================================

evaluated_path, evaluated_length, evaluated_reward, evaluated_goal = \
    generate_path(policy)


# ============================================================
# OPTIMAL POLICY
# ============================================================

optimal_path, optimal_length, optimal_reward, optimal_goal = \
    generate_path(optimal_policy)


# ============================================================
# RANDOM PATH
# ============================================================

random_path, random_length, random_reward, random_goal = \
    generate_path(random_policy)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n========================================")
print("TASK 4: OPTIMAL PATH ANALYSIS")
print("========================================")

print("\nRandom Policy:")
print("Policy:", random_policy)
print("Path:", " -> ".join(random_path))
print("Path Length:", random_length)
print("Total Reward:", random_reward)
print("Goal Reached:", "Yes" if random_goal else "No")


print("\nEvaluated Policy:")
print("Path:", " -> ".join(evaluated_path))
print("Path Length:", evaluated_length)
print("Total Reward:", evaluated_reward)
print("Goal Reached:", "Yes" if evaluated_goal else "No")


print("\nOptimal Policy:")
print("Path:", " -> ".join(optimal_path))
print("Path Length:", optimal_length)
print("Total Reward:", optimal_reward)
print("Goal Reached:", "Yes" if optimal_goal else "No")


# ============================================================
# GRID DISPLAY
# ============================================================

print("\n========================================")
print("GRIDWORLD")
print("========================================")

print("""
+-----+-----+-----+
| S1  | S2  | S3  |
+-----+-----+-----+
| S4  | S5  | G   |
+-----+-----+-----+
""")