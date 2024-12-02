import networkx as nx
import matplotlib.pyplot as plt

# Define the Pacman board
pacman_board = [
    ['A', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D'],
    ['D', 'W', 'W', 'W', 'D', 'W', 'W', 'W', 'D'],
    ['D', 'W', 'D', 'D', 'D', 'D', 'D', 'W', 'D'],
    ['D', 'D', 'D', 'W', 'E', 'W', 'D', 'D', 'D'],
    ['D', 'W', 'D', 'W', 'G', 'W', 'D', 'W', 'D'],
    ['D', 'W', 'D', 'D', 'W', 'D', 'D', 'W', 'D'],
    ['D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D']
]

# Create a bidirectional graph
G = nx.DiGraph()

# Define actions and corresponding rewards
actions = {'U': -1, 'D': 1, 'L': -1, 'R': 1}

# Define rewards for different cell types
rewards = {'D': 10, 'G': -50, 'E': 0}

# Iterate through the Pacman board to create graph vertices and edges
for i in range(len(pacman_board)):
    for j in range(len(pacman_board[i])):
        if pacman_board[i][j] != 'W':
            # Add vertex
            G.add_node((i, j), label=pacman_board[i][j])  # Add label to the node

            # Add edges based on possible actions (Up, Down, Left, Right)
            for action, reward in actions.items():
                ni, nj = i, j
                if action == 'U':
                    ni -= 1
                elif action == 'D':
                    ni += 1
                elif action == 'L':
                    nj -= 1
                elif action == 'R':
                    nj += 1

                # Check if the next position is within the bounds and not a wall
                if 0 <= ni < len(pacman_board) and 0 <= nj < len(pacman_board[i]) and pacman_board[ni][nj] != 'W':
                    next_cell_type = pacman_board[ni][nj]
                    reward_value = rewards.get(next_cell_type)

                    # Add directed edge with corresponding reward
                    G.add_edge((i, j), (ni, nj), weight=reward_value, label=f'{action} ({reward_value})')


# Draw the graph
plt.figure(figsize=(25, 25))
pos = {(i, j): (j, -i) for i, row in enumerate(pacman_board) for j, cell in enumerate(row)}
node_labels = nx.get_node_attributes(G, 'label')  # Get node labels
edge_labels = {(src, dest): label for (src, dest, label) in G.edges(data='label')}  # Get edge labels with weights
edge_colors = ['red' if pacman_board[i][j] == 'G' else 'black' for i, row in enumerate(pacman_board) for j, cell in enumerate(row)]
nx.draw(G, pos, with_labels=True, labels=node_labels, font_weight='bold', node_color='lightblue', node_size=2000, edge_color=edge_colors)

# Draw edge labels for both directions
for (src, dest), label in edge_labels.items():
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(src, dest): label}, font_color='black', font_size=15, label_pos=.3,)

plt.show()
