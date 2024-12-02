import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from matplotlib import animation

# Define the Pacman environment
pacman_board = np.array([
    ['A', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D'],
    ['D', 'W', 'W', 'W', 'D', 'W', 'W', 'W', 'D'],
    ['D', 'W', 'D', 'D', 'D', 'D', 'D', 'W', 'D'],
    ['D', 'D', 'D', 'W', 'E', 'W', 'D', 'D', 'D'],
    ['W', 'W', 'D', 'W', 'G', 'W', 'D', 'W', 'D'],
    ['D', 'E', 'D', 'D', 'W', 'D', 'D', 'W', 'D'],
    ['D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D']
])

# Define actions
actions = ['up', 'down', 'left', 'right']

# Initialize Q-table
q_table = np.zeros((7, 9, 4))

# Define parameters
alpha = 1.0  # learning rate
gamma = 1.0  # discount factor
epsilon = 0.5  # exploration-exploitation trade-off

# Initialize Ghost's position
ghost_position = (4, 4)

# Function to display the Q-table
def display_q_table():
    headers = ["state"] + actions
    table_data = []

    for i in range(q_table.shape[0]):
        for j in range(q_table.shape[1]):
            row_values = [f'({i},{j})'] + [f'{q_table[i, j, k]:.2f}' for k in range(q_table.shape[2])]
            table_data.append(row_values)

    print(tabulate(table_data, headers=headers, tablefmt="pretty"))
    print()

# Function to move the Ghost randomly
def move_ghost(ghost_state):

    # Get available actions for the Ghost
    available_actions = actions.copy()

    row, col = ghost_state

    if row == 0:
        available_actions.remove('up')
    elif row == 6:
        available_actions.remove('down')

    if col == 0:
        available_actions.remove('left')
    elif col == 8:
        available_actions.remove('right')

    # Choose a random action for the Ghost
    action = np.random.choice(available_actions)

    # Update Ghost's position based on the chosen action
    ghost_next_state, _ = take_action(ghost_state, action)
    return ghost_next_state

# Function to get the next state and reward based on the action
def take_action(state, action):
    row, col = state

    if action == 'up' and row > 0:
        row -= 1
    elif action == 'down' and row < 6:
        row += 1
    elif action == 'left' and col > 0:
        col -= 1
    elif action == 'right' and col < 8:
        col += 1

    next_state = (row, col)

    # Define rewards and update the environment based on the next state
    if pacman_board[row, col] == 'D':
        reward = 1
    elif pacman_board[row, col] == 'G':
        reward = -5
    elif pacman_board[row, col] == 'E' or pacman_board[row, col] == 'U':
        reward = 0
    else:
        next_state = state
        reward = 0

    return next_state, reward


# Q-learning algorithm
def q_learning(i):
    global q_table
    global pacman_board

    state = (0, 0)  # initial state
    total_reward = 0

    ghost_state = (4,4)

    # Reset Pacman board for a new episode
    pacman_board = np.array([
        ['A', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D'],
        ['D', 'W', 'W', 'W', 'D', 'W', 'W', 'W', 'D'],
        ['D', 'W', 'D', 'D', 'D', 'D', 'D', 'W', 'D'],
        ['D', 'D', 'D', 'W', 'E', 'W', 'D', 'D', 'D'],
        ['W', 'W', 'D', 'W', 'G', 'W', 'D', 'W', 'D'],
        ['D', 'E', 'D', 'D', 'W', 'D', 'D', 'W', 'D'],
        ['D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D']
    ])

    cp_board = pacman_board.copy()

    while state != (6, 8):  # goal state
        # Move Ghost
        ghost_next_state = move_ghost(ghost_state)

        if cp_board[ghost_state[0], ghost_state[1]] == 'G':
            pacman_board[ghost_state[0], ghost_state[1]] = 'E'
        else:
            pacman_board[ghost_state[0], ghost_state[1]] = cp_board[ghost_state[0], ghost_state[1]]
        pacman_board[ghost_next_state[0], ghost_next_state[1]] = 'G'

        ghost_state = ghost_next_state

        # Choose action using epsilon-greedy strategy
        if np.random.rand() < epsilon:
            action = np.random.choice(actions)
        else:
            action = actions[np.argmax(q_table[state[0], state[1]])]

        # Take action and observe next state and reward
        next_state, reward = take_action(state, action)

        pacman_board[state[0], state[1]] = 'U'  # Convert Agent's previous position to Empty cell
        pacman_board[next_state[0], next_state[1]] = 'A'

        # Update Q-value
        q_table[state[0], state[1], actions.index(action)] += \
            alpha * (reward + gamma * np.max(q_table[next_state[0], next_state[1]]) - q_table[state[0], state[1], actions.index(action)])

        state = next_state
        total_reward += reward

    plt.clf()

    # Display the Pacman board
    for row in range(7):
        for col in range(9):
            if pacman_board[row, col] == 'W':
                plt.plot(col, row, 'ks', markersize=20)
            elif pacman_board[row, col] == 'D':
                plt.plot(col, row, 'bo', markersize=20)
            elif pacman_board[row, col] == 'G':
                plt.plot(col, row, 'ro', markersize=20)
            elif pacman_board[row, col] == 'E':
                plt.plot(col, row, 'ws', markersize=20)
            elif pacman_board[row, col] == 'U':
                plt.plot(col, row, 'yo', markersize=20)
            elif pacman_board[row, col] == 'A':
                plt.plot(col, row, 'go', markersize=20)

    # Display episode and total reward
    print(f"Episode {i + 1}, Total Reward: {total_reward}")
    print(pacman_board)
    print()
    pacman_board[0][0] = 'E'
    display_q_table()
    plt.title(f"Episode: {i + 1}")

# Display the Pacman board using animation
fig, ax = plt.subplots()
ani = animation.FuncAnimation(fig, q_learning, frames=4, interval=500, repeat=False)

# For saving the animation to a file (e.g., GIF)
ani.save('modified_pacman_animation.gif', writer='pillow')