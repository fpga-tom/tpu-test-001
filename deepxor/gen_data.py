import random
import csv

solved = [1, 1, 1, 1, 0, 0, 0, 0]
num_actions = len(solved) + 1

def apply_action(state, action):
    state = [i for i in state]
    if action < len(solved):
        state[action] ^= 1
    return state

def reward(state):
    if all([_solved == _state for _solved, _state in zip(solved, state)]):
        return 1
    return -1

def generate(k):
    result = [(solved, 0)]
    current = solved
    for i in range(0,k):
        state = apply_action(current, random.randint(0,num_actions-1))
        result.append((state, i+1))
        current = state
    return result

def generate_all(l, k):
    with open('x_input.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(0, l):
            for row in generate(k):
                writer.writerow(row[0] + [row[1]])
                for a in range(0, num_actions):
                    writer.writerow(apply_action(row[0], a) + [row[1]])

generate_all(100,100)
