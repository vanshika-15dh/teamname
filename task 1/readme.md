# RoboGambit Engine – Task 1

## Introduction

In this task, we built a chess engine capable of playing a 6×6 chess variant. The engine receives a board configuration and determines a strong move for the player whose turn it is.

The overall approach is to generate all possible moves, explore how the game might continue after those moves, and select the move that leads to the most favorable board position.

Since the number of possible game states grows extremely quickly, exploring every possibility would take too long. To make the engine practical, we used several techniques that allow it to focus only on promising branches of the search while ignoring clearly weaker options.

## Board Representation

The chess board is stored as a NumPy array of length 36, representing the 6×6 board. Each number represents a piece on the board:

| Piece  | White ID | Black ID |
|--------|----------|----------|
| Pawn   | 1        | 6        |
| Knight | 2        | 7        |
| Bishop | 3        | 8        |
| Queen  | 4        | 9        |
| King   | 5        | 10       |

Instead of using a 2‑D matrix, we store the board in a flattened format, which simplifies indexing and speeds up board updates.

Example initialization:

```python
import numpy as np

board = np.zeros(36, dtype=int)
board = 4   # white queen
board = 10 # black king

This compact representation makes the engine faster when accessing or modifying squares.

##Move Representation
To keep the engine efficient, we represent every move using a single integer. The source and destination coordinates are packed together using bit operations.

Example encoding:

def encode_move(sr, sc, dr, dc):
    return sr | (sc << 3) | (dr << 6) | (dc << 9)

Decoding a move:
def decode_move(move):
    sr =  move        & 7
    sc = (move >> 3)  & 7
    dr = (move >> 6)  & 7
    dc = (move >> 9)  & 7
    return sr, sc, dr, dc

Using integers instead of complex objects helps reduce memory usage and speeds up comparisons inside the search.

## Exploring Possible Moves
To choose the best move, the engine first generates all legal moves for the current player. For each move:

The move is temporarily applied to the board.

The program examines possible responses by the opponent.

This process continues several steps into the future.

Example of applying a move on a flat 36‑element board:
def make_move(board, move):
    sr, sc, dr, dc = decode_move(move)
    src = sr * 6 + sc
    dst = dr * 6 + dc

    piece = board[src]
    board[src] = 0
    board[dst] = piece
During this exploration, the engine keeps track of the best position found so far. If it becomes clear that a certain path cannot lead to a better result, the engine stops exploring that path (similar to alpha–beta pruning).

## Gradual Deepening of Search
Instead of immediately searching very deep positions, the engine analyzes the game in stages. It first looks a few moves ahead and then gradually increases the depth.

Example structure:
def search(board, depth):
    # placeholder for actual search implementation
    best_move = None
    best_score = -10**9
    # ...
    return best_move

def iterative_search(board, max_depth):
    best_move = None
    depth = 1
    while depth <= max_depth:
        best_move = search(board, depth)
        depth += 1
    return best_move
This approach ensures that the engine always has a valid move ready, even if the time limit is reached before the deepest search finishes.

## Position Evaluation
When the engine reaches the end of a search branch, it estimates how favorable the board position is. The evaluation mainly considers:

Material balance

Piece mobility

Pawn structure

Piece placement

Example of a simple evaluation idea:
piece_values = {
    1: 100, 2: 320, 3: 330, 4: 900, 5: 20000,
    6: -100, 7: -320, 8: -330, 9: -900, 10: -20000,
}

def evaluate(board):
    score = 0
    for piece in board:
        score += piece_values.get(int(piece), 0)
    return score

This score helps the engine decide which positions are stronger from the current player’s perspective.

Avoiding Repeated Calculations
During the search, the same board position may appear multiple times through different move sequences. To prevent repeating the same work, the engine stores information about positions that were already analyzed. If the same position appears again later, the stored result can be reused instead of recomputing it.

Example idea:
visited_positions = {}

def search_with_cache(board, depth):
    position_hash = hash(board.tobytes())
    if position_hash in visited_positions and visited_positions[position_hash] >= depth:
        return visited_positions[position_hash][1]

    score = search(board, depth)
    visited_positions[position_hash] = (depth, score)
    return score
This significantly improves performance when the search goes deeper.

## Pawn Promotion Rule
The promotion rule used in this variant differs slightly from standard chess. A pawn can only promote to a piece that has already been captured earlier in the game.

Example logic:
def handle_promotion(board, index, captured_pieces, is_white):
    pawn = board[index]
    last_rank = (index // 6 == 5) if is_white else (index // 6 == 0)

    if pawn in (1, 6) and last_rank and captured_pieces:
        promote_to = captured_pieces.pop()  # choose from previously captured pieces
        board[index] = promote_to

This rule makes promotion more dynamic and depends on how the game has progressed.

## Testing the Engine
To verify the engine’s behavior, we tested it with two types of starting boards.

Fixed Starting Position
Useful for debugging and consistent testing.
import numpy as np

fixed_board = np.array([
    2, 3, 4, 5, 3, 2,
    1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
    6, 6, 6, 6, 6, 6,
    7, 8, 9,10, 8, 7,
], dtype=int)

## Randomized Starting Position
Back rank pieces are shuffled while keeping bishops on opposite colors.
import numpy as np
import random

def random_back_rank(white=True):
    # Example pool: [N, B, Q, K, B, N]
    pieces =  if white else[2][3][4][5][6][7][8][9]
    random.shuffle(pieces)
    return pieces

back = random_back_rank(white=True)
board = np.zeros(36, dtype=int)
for c, p in enumerate(back):
    board[c] = p

Randomized setups ensure the engine handles a wide range of positions.

## Final Move Output
Once the engine finishes its analysis, it returns the selected move in the format:
<piece_id>:<source_square>-><destination_square>

Example output:
1:A2->A3
This means a white pawn moves from A2 to A3.

## Conclusion
In this project, we implemented a chess engine capable of playing a 6×6 chess variant. The engine generates moves, explores possible continuations of the game, evaluates board positions, and selects a move that appears most promising.

Through efficient board representation, careful exploration of game states, and reuse of previously analyzed positions, the engine is able to analyze deeper positions within a limited time.

Working on this project helped us understand how strategic decision-making in board games can be translated into computational algorithms and optimized for performance.

undefined
