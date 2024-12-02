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
    ['D', 'W', 'D', 'W', 'G', 'W', 'D', 'W', 'D'],
    ['D', 'W', 'D', 'D', 'W', 'D', 'D', 'W', 'D'],
    ['D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D']
])

# Define actions
actions = ['up', 'down', 'left', 'right']

# Initialize Q-table
q_table = np.zeros((7, 9, 4))

# Define parameters
alpha = 0.1  # learning rate
gamma = 0.9  # discount factor
epsilon = 0.5  # exploration-exploitation trade-off

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

    # Reset Pacman board for a new episode
    pacman_board = np.array([
        ['A', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D'],
        ['D', 'W', 'W', 'W', 'D', 'W', 'W', 'W', 'D'],
        ['D', 'W', 'D', 'D', 'D', 'D', 'D', 'W', 'D'],
        ['D', 'D', 'D', 'W', 'E', 'W', 'D', 'D', 'D'],
        ['D', 'W', 'D', 'W', 'G', 'W', 'D', 'W', 'D'],
        ['D', 'W', 'D', 'D', 'W', 'D', 'D', 'W', 'D'],
        ['D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D']
    ])

    while state != (6, 8):  # goal state
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
    plt.title(f"Episode: {i + 1}")
    # Display episode and total reward
    pacman_board[0][0] = 'E'
    print(f"Episode {i + 1}, Total Reward: {total_reward}")
    print(pacman_board)
    print()
    display_q_table()


# Display the Pacman board using animation
fig, ax = plt.subplots()
ani = animation.FuncAnimation(fig, q_learning, frames=20, interval=1000, repeat=False)

# For saving the animation to a file (e.g., GIF)
ani.save('pacman_animation.gif', writer='pillow')