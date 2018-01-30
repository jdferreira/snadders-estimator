"""
Compute the expected number of rolls to win a snakes and ladders game.
"""

import numpy as np


def get_probabilities(size, snadders=None, dice_size=6, transitive=False, exact=False):
    """
    Generate the transition probabilities matrix for a board, given:
    
    - its `size` (the number of tiles)
    - the snakes and ladders transitions (a python dictionary)
    - the number of sides in the dice
    - whether a snadder ending where another begins makes the player jump in a
      `transitive` manner or not (note that if the board contains a cycle of
      snadders and the function is called with the transitive parameter set to
      `True`, an infinite cycle will be reached)
    - whether the end move must be `exact` or any value greater than or equal to
      the required amount is enough to win. If `exact` is `True`, a roll of more
      than the required amount makes the player stay in the same tile
    
    The tiles are numbered from `1` to `size`, with an additional tile `0` being
    used for the starting tile (thus, rolling a 2 in the first round moves the
    player to tile 2).
    
    The function returns a dictionary where keys are pairs `(a, b)` and the
    values are the probability of transitioning from tile `a` to tile `b`.
    """
    
    snadders = snadders or {}
    
    # Any probability is 0 by default. We have `size + 1` states, counting the
    # starting tile
    result = {(i, j): 0.0 for i in range(size + 1) for j in range(size + 1)}
    
    for idx in range(size):
        # We're in spot `idx`. We can roll any number between 1 and the
        # "sidedness" of the dice.
        
        # Note that there is no transition from the end state to other states,
        # and so we do not need to consider the case `idx == size`.
        
        for roll in range(1, dice_size + 1):
            end = idx + roll
            
            # If we transitively cross all snadders, do so while one exists;
            # otherwise, just jump once.
            if transitive:
                while end in snadders:
                    end = snadders[end]
            else:
                if end in snadders:
                    end = snadders[end]
            
            # If we went past the end tile, we need to know if this means we won
            # or we don't move
            if end > size:
                if exact:
                    end = idx
                else:
                    end = size
            
            result[idx, end] += 1.0 / dice_size
    
    return result


def make_matrices(probs, size):
    """
    Converts a transition probabilities dictionary (returned from 
    `get_probabilities`) into a pair of numpy matrices that can be used to
    solve the problem at hand. In particular, the two matrices are `A` and `B`,
    which, in the form $A \\cdot X = B$, where row `n` of that system corresponds
    to the equation:
    
        $e_n = 1 + \\sum_k{P(n \\leftarrow k) \\cdot e_k}$
    
    In this, $e_n$ is the expected number of rolls to win if we're at tile `n`
    and $P(n \\leftarrow k)$ is the probability of transitioning from tile `n` to
    tile `k`, which are given by the `probs` parameter.
    """
    
    a = np.zeros((size + 1, size + 1))
    for i in range(size + 1):
        for j in range(size + 1):
            value = probs[i, j]
            if i == j:
                value -= 1
            a[i, j] = value
    
    b = np.zeros((size + 1, 1)) - 1
    b[size] = 0
    
    # print(np.concatenate((a, b), axis=1))
    return a, b


def solve(board_size, snadders, dice_size):
    """
    Create the transition probabilities for the board of the given size and the
    the given snadders. Use a dice of a specific number of sides to roll and
    move.
    """
    
    probs = get_probabilities(board_size, snadders, dice_size)
    a, b = make_matrices(probs, board_size)
    solution = np.linalg.solve(a, b)
    
    return float(solution[0])


def main():
    """
    Entry point
    """
    
    # # Small boards for testing purposes
    # print(solve(9, {1: 4, 7: 2}, 6))
    # print(solve(3, {1: 2}, 2))
    
    # print(solve(6, None, 3))
    # print(solve(6, {2: 4}, 3))
    
    # Board from 42:52 in https://www.youtube.com/watch?v=1v_E18xU3ok
    board_size = 100
    snadders = {
        12: 25,
        22: 32,
        29: 90,
        34: 14,
        40: 2,
        99: 44,
    }
    dice_size = 6
    estimate = solve(board_size, snadders, dice_size)
    print(estimate)
    
    # Board from 48:18 in https://www.youtube.com/watch?v=1v_E18xU3ok
    board_size = 100
    snadders = {
        13: 50,
        24: 10,
        28: 88,
        78: 59,
        81: 93,
        92: 22
        
    }
    dice_size = 6
    estimate = solve(board_size, snadders, dice_size)
    print(estimate)


if __name__ == '__main__':
    main()
