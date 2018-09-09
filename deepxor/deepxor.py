import copy

solved = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1]
len_solved = len(solved)
num_actions = len_solved

def apply_action(state, action):
    state = [i for i in state]
    if action < len(solved):
        state[action] ^= 1
    return state

def reward(state):
    if all([_solved == _state for _solved, _state in zip(solved, state)]):
        return 1
    return -1

class Position():
    def __init__(self, n=0):
        self.n = n
        self.state = [0 for x in range(0, len_solved)]
        self.to_play = 1

    def play_move(self, c):

        pos = copy.deepcopy(self)
        pos.state = apply_action(pos.state, c)
        pos.n += 1

        return pos

    def score(self):
        return reward(self.state)

    def all_legal_moves(self):
        return num_actions

