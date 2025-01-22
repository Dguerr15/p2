from mcts_node import MCTSNode
from p2_t3 import Board
from random import choice
from math import sqrt, log

num_nodes = 1000
explore_faction = 2.

def traverse_nodes(node: MCTSNode, board: Board, state, bot_identity: int):
    """ Traverses the tree until the end criterion are met.
    e.g. find the best expandable node (node with untried action) if it exist,
    or else a terminal node

    Args:
        node:       A tree node from which the search is traversing.
        board:      The game setup.
        state:      The state of the game.
        identity:   The bot's identity, either 1 or 2

    Returns:
        node: A node from which the next stage of the search can proceed.
        state: The state associated with that node
    """
    while not node.untried_actions and node.child_nodes:
        node = max(node.child_nodes.values(), 
                  key=lambda child: ucb(child, is_opponent=(bot_identity != board.current_player(state))))
        state = board.next_state(state, node.parent_action)
    return node, state

def expand_leaf(node: MCTSNode, board: Board, state):
    """ Adds a new leaf to the tree by creating a new child node for the given node (if it is non-terminal).

    Args:
        node:   The node for which a child will be added.
        board:  The game setup.
        state:  The state of the game.

    Returns:
        node: The added child node
        state: The state associated with that node
    """
    action = node.untried_actions.pop()
    next_state = board.next_state(state, action)
    child_node = MCTSNode(parent=node, parent_action=action, action_list=board.legal_actions(next_state))
    node.child_nodes[action] = child_node
    return child_node, next_state

def rollout(board: Board, state):
    """ Given the state of the game, the rollout plays out the remainder randomly but with some basic heuristics.
    """
    curr_state = state
    while not board.is_ended(curr_state):
        actions = board.legal_actions(curr_state)
        if not actions:
            break
            
        # Create weighted list with stronger preferences
        weighted_actions = []
        for action in actions:
            _, _, r, c = action
            if (r, c) == (1, 1):  # Center (highest priority)
                weighted_actions.extend([action] * 5)  # Increased from 3 to 5
            elif (r in [0, 2] and c in [0, 2]):  # Corners
                weighted_actions.extend([action] * 3)  # Increased from 2 to 3
            else:  # Edges
                weighted_actions.append(action)
                
        # Choose randomly from weighted list
        action = choice(weighted_actions)
        curr_state = board.next_state(curr_state, action)
        
    return curr_state

def backpropagate(node: MCTSNode|None, won: bool):
    """ Navigates the tree from a leaf node to the root, updating the win and visit count of each node along the path.

    Args:
        node:   A leaf node.
        won:    An indicator of whether the bot won or lost the game.
    """
    while node is not None:
        node.visits += 1
        node.wins += won
        node = node.parent

def ucb(node: MCTSNode, is_opponent: bool):
    """ Calculates the UCB value for the given node from the perspective of the bot

    Args:
        node:   A node.
        is_opponent: A boolean indicating whether or not the last action was performed by the MCTS bot
    Returns:
        The value of the UCB function for the given node
    """
    if node.visits == 0:
        return float('inf')
    win_rate = (1 - node.wins / node.visits) if is_opponent else (node.wins / node.visits)
    return win_rate + explore_faction * sqrt(log(node.parent.visits) / node.visits)

def get_best_action(root_node: MCTSNode):
    """ Selects the best action from the root node in the MCTS tree

    Args:
        root_node:   The root node
    Returns:
        action: The best action from the root node
    """
    return max(root_node.child_nodes.items(), key=lambda item: item[1].visits)[0]

def is_win(board: Board, state, identity_of_bot: int):
    # checks if state is a win state for identity_of_bot
    outcome = board.points_values(state)
    assert outcome is not None, "is_win was called on a non-terminal state"
    return outcome[identity_of_bot] == 1

def think(board: Board, current_state):
    """ Performs MCTS by sampling games and calling the appropriate functions to construct the game tree.

    Args:
        board:  The game setup.
        current_state:  The current state of the game.

    Returns:    The action to be taken from the current state
    """
    bot_identity = board.current_player(current_state)
    root_node = MCTSNode(parent=None, parent_action=None, action_list=board.legal_actions(current_state))

    for _ in range(num_nodes):
        state = current_state
        node = root_node
        
        # Traverse the tree
        node, state = traverse_nodes(node, board, state, bot_identity)
        
        # Expand the leaf node
        if node.untried_actions:
            node, state = expand_leaf(node, board, state)
            
        # Simulate a game
        final_state = rollout(board, state)
        
        # Backpropagate the result
        won = is_win(board, final_state, bot_identity)
        backpropagate(node, won)

    # Return an action, typically the most frequently used action (from the root)
    best_action = get_best_action(root_node)
    
    print(f"Action chosen: {best_action}")
    return best_action
