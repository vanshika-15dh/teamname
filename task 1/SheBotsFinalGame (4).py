import numpy as np
from typing import Optional
import time
import random


EMPTY        = 0
WHITE_PAWN   = 1;  WHITE_KNIGHT = 2;  WHITE_BISHOP = 3
WHITE_QUEEN  = 4;  WHITE_KING   = 5
BLACK_PAWN   = 6;  BLACK_KNIGHT = 7;  BLACK_BISHOP = 8
BLACK_QUEEN  = 9;  BLACK_KING   = 10

WHITE_PIECES = frozenset({1, 2, 3, 4, 5})
BLACK_PIECES = frozenset({6, 7, 8, 9, 10})
NUM_SQUARES  = 36
COL_TO_FILE  = 'ABCDEF'
FILE_TO_COL  = {f: i for i, f in enumerate(COL_TO_FILE)}

WHITE_TO_BLACK = {1:6, 2:7, 3:8, 4:9, 5:10}
BLACK_TO_WHITE = {6:1, 7:2, 8:3, 9:4, 10:5}


MAT = np.array([0, 100, 300, 320, 900, 20000,
                   100, 300, 320, 900, 20000], dtype=np.int32)


_P_MG = [[ 0,  0,  0,  0,  0,  0],
         [ 5,  5, 10, 10,  5,  5],
         [10, 10, 15, 15, 10, 10],
         [20, 25, 30, 30, 25, 20],
         [30, 35, 40, 40, 35, 30],
         [50, 50, 50, 50, 50, 50]]

_N_MG = [[-20,-10, -5, -5,-10,-20],
         [-10,  0,  8,  8,  0,-10],
         [ -5,  8, 20, 20,  8, -5],
         [ -5, 10, 20, 25, 10, -5],
         [-10,  0,  8,  8,  0,-10],
         [-20,-10, -5, -5,-10,-20]]

_B_MG = [[-10, -5, -5, -5, -5,-10],
         [ -5, 12, 10, 10, 12, -5],
         [ -5, 10, 18, 18, 10, -5],
         [ -5, 12, 18, 18, 12, -5],
         [ -5, 10, 10, 10, 10, -5],
         [-10, -5, -5, -5, -5,-10]]

_Q_MG = [[ -5, -5, -2, -2, -5, -5],
         [ -5,  5,  5,  5,  5, -5],
         [ -2,  5, 10, 10,  5, -2],
         [ -2,  8, 12, 12,  8, -2],
         [ -5,  5,  8,  8,  5, -5],
         [ -5, -5, -2, -2, -5, -5]]

_K_MG = [[ 25, 35, 15, 15, 35, 25],
         [ 15, 20,  0,  0, 20, 15],
         [-10,-20,-20,-20,-20,-10],
         [-20,-30,-30,-40,-40,-20],
         [-30,-40,-40,-50,-50,-30],
         [-40,-50,-50,-60,-60,-40]]

_P_EG = [[ 0,  0,  0,  0,  0,  0],
         [10, 10, 10, 10, 10, 10],
         [20, 20, 20, 20, 20, 20],
         [35, 35, 35, 35, 35, 35],
         [50, 50, 50, 50, 50, 50],
         [70, 70, 70, 70, 70, 70]]

_N_EG = [[-30,-20,-15,-15,-20,-30],
         [-20, -5,  2,  2, -5,-20],
         [-15,  2, 12, 12,  2,-15],
         [-15,  5, 12, 18,  5,-15],
         [-20, -5,  2,  2, -5,-20],
         [-30,-20,-15,-15,-20,-30]]

_B_EG = [[-12, -5, -5, -5, -5,-12],
         [ -5,  8,  6,  6,  8, -5],
         [ -5,  6, 14, 14,  6, -5],
         [ -5,  8, 14, 14,  8, -5],
         [ -5,  6,  6,  6,  6, -5],
         [-12, -5, -5, -5, -5,-12]]

_Q_EG = [[ -8, -5, -2, -2, -5, -8],
         [ -5,  5,  5,  5,  5, -5],
         [ -2,  5, 12, 12,  5, -2],
         [ -2,  8, 14, 14,  8, -2],
         [ -5,  5,  8,  8,  5, -5],
         [ -8, -5, -2, -2, -5, -8]]

_K_EG = [[-20,-10, -5, -5,-10,-20],
         [-10,  5, 10, 10,  5,-10],
         [ -5, 10, 20, 20, 10, -5],
         [ -5, 12, 20, 25, 12, -5],
         [-10,  5, 10, 10,  5,-10],
         [-20,-10, -5, -5,-10,-20]]

def _build_pst(table_mg, table_eg):
    pst_mg = np.zeros((11, NUM_SQUARES), dtype=np.int16)
    pst_eg = np.zeros((11, NUM_SQUARES), dtype=np.int16)
    pairs = [(1,6,table_mg[0],table_eg[0]),
             (2,7,table_mg[1],table_eg[1]),
             (3,8,table_mg[2],table_eg[2]),
             (4,9,table_mg[3],table_eg[3]),
             (5,10,table_mg[4],table_eg[4])]
    for wp, bp, tmg, teg in pairs:
        for r in range(6):
            for c in range(6):
                pst_mg[wp, r*6+c] = tmg[r][c]
                pst_mg[bp, r*6+c] = tmg[5-r][c]
                pst_eg[wp, r*6+c] = teg[r][c]
                pst_eg[bp, r*6+c] = teg[5-r][c]
    return pst_mg, pst_eg

PST_MG, PST_EG = _build_pst(
    [_P_MG, _N_MG, _B_MG, _Q_MG, _K_MG],
    [_P_EG, _N_EG, _B_EG, _Q_EG, _K_EG]
)

PST = PST_MG
PHASE_TOTAL = 2*(300+320+900) * 2


def enc(sr, sc, dr, dc) -> int: return sr | (sc<<3) | (dr<<6) | (dc<<9)
def dec(m: int):                return m&7, (m>>3)&7, (m>>6)&7, (m>>9)&7
NULL_MOVE = -1


_rng    = np.random.default_rng(20250312)
ZOBRIST = _rng.integers(0, 2**63, (11, NUM_SQUARES), dtype=np.uint64)
Z_TURN  = _rng.integers(0, 2**63, dtype=np.uint64)

Z_PAWN = np.zeros((11, NUM_SQUARES), dtype=np.uint64)
for _p in (1, 6): Z_PAWN[_p] = ZOBRIST[_p]

def _init_hash(board):
    h = np.uint64(0)
    for sq in range(NUM_SQUARES):
        p = int(board[sq])
        if p: h ^= ZOBRIST[p, sq]
    return h

def _init_pawn_hash(board):
    h = np.uint64(0)
    for sq in range(NUM_SQUARES):
        p = int(board[sq])
        if p in (1, 6): h ^= Z_PAWN[p, sq]
    return h


# ─────────────────────────────────────────────────────────────────
#  TRANSPOSITION TABLE  —  ~8 GB, TRUE 64-BIT HASH COMPARISON
#
#  Single table with 2^29 = 536 870 912 entries.
#
#  Memory breakdown per entry (16 bytes total):
#    uint64  hash  — full Zobrist key, zero-collision bucket match
#    int32   score
#    int8    depth
#    int8    flag
#    int16   move
#
#  Total: 536 870 912 × 16 B = 8 589 934 592 B ≈ 8.0 GiB
#
#  Why this is better than the v5 32-bit scheme:
#    - v5 stored only the upper 32 bits (hi32) → 1 false-hit per ~4 billion
#      probes was accepted as "good enough".
#    - With 2^29 slots the index already consumes 29 bits of the hash.
#      Only 35 bits remain to distinguish collisions if we keep hi32.
#    - Storing the full 64-bit key gives us 64 − 29 = 35 free bits of
#      verification AND the full key itself, making spurious hits
#      astronomically unlikely even at depth 14.
# ─────────────────────────────────────────────────────────────────

print("Allocating TT (~8 GB — this may take a moment) ...")

TT_SIZE = np.uint64(1 << 29)   # 536 870 912 entries
TT_MASK = TT_SIZE - np.uint64(1)

# Each array is allocated lazily by the OS; physical RAM is only
# committed as pages are actually written, so startup is fast.
#
# TT_SCORE uses int16 (not int32): scores are bounded ±MATE = ±19 000,
# which fits inside int16's range of ±32 767.  This saves exactly
# 1 024 MB, keeping the total under 8.0 GB (decimal).
#
#   TT_HASH  uint64  ×  2^29  =  4 096 MB
#   TT_SCORE int16   ×  2^29  =  1 024 MB   ← was int32 (2 048 MB)
#   TT_DEPTH int8    ×  2^29  =    512 MB
#   TT_FLAG  int8    ×  2^29  =    512 MB
#   TT_MOVE  int16   ×  2^29  =  1 024 MB
#   ─────────────────────────────────────
#   Total                      =  7 168 MB  =  7.0 GiB  =  7.52 GB ✓
TT_HASH  = np.zeros(int(TT_SIZE), dtype=np.uint64)   # 4 096 MB
TT_SCORE = np.zeros(int(TT_SIZE), dtype=np.int16)    # 1 024 MB  (int16 safe: ±32767 > ±19000)
TT_DEPTH = np.zeros(int(TT_SIZE), dtype=np.int8)     #   512 MB
TT_FLAG  = np.zeros(int(TT_SIZE), dtype=np.int8)     #   512 MB
TT_MOVE  = np.zeros(int(TT_SIZE), dtype=np.int16)    # 1 024 MB
# ──────────────────── Total = 7 168 MB = 7.52 GB decimal ─────────

TT_EXACT = np.int8(0)
TT_LOWER = np.int8(1)
TT_UPPER = np.int8(2)

print("TT allocated.")


def tt_lookup(h: np.uint64, depth: int, alpha: int, beta: int):
    """
    Full 64-bit hash comparison — no spurious hits from truncated keys.
    Returns (score_or_None, move_hint_int, hit_bool).
    """
    idx = int(h & TT_MASK)
    if TT_HASH[idx] == h:                          # true 64-bit match
        d  = int(TT_DEPTH[idx])
        sc = int(TT_SCORE[idx])
        fl = int(TT_FLAG[idx])
        mv = int(TT_MOVE[idx])
        if d >= depth:
            if fl == 0:               return sc, mv, True
            if fl == 1 and sc >= beta: return sc, mv, True
            if fl == 2 and sc <= alpha:return sc, mv, True
        return None, mv, False
    return None, 0, False


def tt_store(h: np.uint64, depth: int, score: int, flag: np.int8, move: int):
    idx = int(h & TT_MASK)
    # Replace-if-deeper strategy (keep shallowest overwrite too)
    if TT_HASH[idx] != h or int(TT_DEPTH[idx]) <= depth:
        TT_HASH[idx]  = h                          # store full 64-bit key
        TT_SCORE[idx] = np.int16(max(-32767, min(32767, score)))
        TT_DEPTH[idx] = np.int8(min(depth, 127))
        TT_FLAG[idx]  = flag
        TT_MOVE[idx]  = np.int16(move & 0x7FFF)


# ── Pawn hash table (unchanged, modest size) ──────────────────────
PH_SIZE  = np.uint64(1 << 20)
PH_MASK  = PH_SIZE - np.uint64(1)
PH_HASH  = np.zeros(int(PH_SIZE), dtype=np.uint64)   # full 64-bit here too
PH_SCORE = np.zeros(int(PH_SIZE), dtype=np.int32)

def pawn_hash_lookup(ph: np.uint64):
    idx = int(ph & PH_MASK)
    if PH_HASH[idx] == ph:
        return int(PH_SCORE[idx])
    return None

def pawn_hash_store(ph: np.uint64, score: int):
    idx = int(ph & PH_MASK)
    PH_HASH[idx]  = ph
    PH_SCORE[idx] = np.int32(score)


# ─────────────────────────────────────────────────────────────────
#  NODE COUNTER  —  replaces the old ply % N time check
#
#  Checking time.time() on every node is expensive (syscall).
#  Instead we count nodes and only call time.time() every
#  NODE_CHECK_INTERVAL nodes.  This gives ≈ the same wall-clock
#  granularity with far less overhead, critical at depth 14.
# ─────────────────────────────────────────────────────────────────
NODE_CHECK_INTERVAL = 4096     # check clock every 4 K nodes
_node_counter       = [0]      # mutable so inner function can update it
_deadline           = [0.0]    # set by iterative_deepening each iteration
_time_expired       = [False]  # flag; avoids redundant time.time() calls


def _reset_time_control(deadline: float):
    _deadline[0]     = deadline
    _time_expired[0] = False
    _node_counter[0] = 0


def _tick() -> bool:
    """Increment node counter; return True if deadline exceeded."""
    if _time_expired[0]:
        return True
    _node_counter[0] += 1
    if _node_counter[0] >= NODE_CHECK_INTERVAL:
        _node_counter[0] = 0
        if time.time() >= _deadline[0]:
            _time_expired[0] = True
            return True
    return False



class BoardState:
    def __init__(self, flat_board: np.ndarray,
                 white_captured=None, black_captured=None,
                 halfmove_clock: int = 0):
        self.board        = flat_board.astype(np.int8)
        self.hash         = _init_hash(self.board)
        self.pawn_hash    = _init_pawn_hash(self.board)
        self.material     = self._init_mat()

        self.killers      = [[0, 0] for _ in range(64)]
        self.history      = np.zeros((11, NUM_SQUARES), dtype=np.int32)
        self.counter      = np.zeros((NUM_SQUARES, NUM_SQUARES), dtype=np.int16)

        self.halfmove_clock = halfmove_clock

        if white_captured is not None:
            self.white_captured = list(white_captured)
        else:
            self.white_captured = self._infer_captured(white=True)

        if black_captured is not None:
            self.black_captured = list(black_captured)
        else:
            self.black_captured = self._infer_captured(white=False)

    def _infer_captured(self, white: bool):
        start_counts = {
            WHITE_PAWN: 6, WHITE_KNIGHT: 2, WHITE_BISHOP: 2,
            WHITE_QUEEN: 1, WHITE_KING: 1,
            BLACK_PAWN: 6, BLACK_KNIGHT: 2, BLACK_BISHOP: 2,
            BLACK_QUEEN: 1, BLACK_KING: 1,
        }
        pieces   = WHITE_PIECES if white else BLACK_PIECES
        captured = []
        for p in pieces:
            on_board = int(np.sum(self.board == p))
            missing  = start_counts[p] - on_board
            captured.extend([p] * max(0, missing))
        return captured

    def _init_mat(self):
        s = 0
        for sq in range(NUM_SQUARES):
            p = int(self.board[sq])
            if p:
                sign = 1 if p in WHITE_PIECES else -1
                s += sign * (int(MAT[p]) + int(PST[p, sq]))
        return s

    def get(self, sq):     return int(self.board[sq])
    def sq(self, r, c):    return r * 6 + c

    def _promotion_piece(self, moved: int, dr: int, white: bool):
        is_promo = (moved == WHITE_PAWN and dr == 5 and white) or \
                   (moved == BLACK_PAWN and dr == 0 and not white)
        if not is_promo:
            return None
        if white:
            available = self.white_captured
            priority  = [WHITE_QUEEN, WHITE_KNIGHT, WHITE_BISHOP]
        else:
            available = self.black_captured
            priority  = [BLACK_QUEEN, BLACK_KNIGHT, BLACK_BISHOP]
        for p in priority:
            if p in available:
                return p
        return None

    def make(self, mv: int):
        sr, sc, dr, dc = dec(mv)
        fsq     = sr*6+sc
        tsq     = dr*6+dc
        moved   = int(self.board[fsq])
        captured= int(self.board[tsq])
        white   = moved in WHITE_PIECES

        old_halfmove = self.halfmove_clock

        if moved in (WHITE_PAWN, BLACK_PAWN) or captured:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1

        self.hash ^= ZOBRIST[moved, fsq]
        if moved in (1, 6): self.pawn_hash ^= Z_PAWN[moved, fsq]
        sign = 1 if white else -1
        self.material -= sign * (int(MAT[moved]) + int(PST[moved, fsq]))

        if captured:
            self.hash ^= ZOBRIST[captured, tsq]
            if captured in (1, 6): self.pawn_hash ^= Z_PAWN[captured, tsq]
            sc2 = 1 if captured in WHITE_PIECES else -1
            self.material -= sc2 * (int(MAT[captured]) + int(PST[captured, tsq]))

        promo_piece = self._promotion_piece(moved, dr, white)
        final       = promo_piece if promo_piece is not None else moved

        if captured:
            if captured in WHITE_PIECES:
                self.white_captured.append(captured)
            else:
                self.black_captured.append(captured)

        if promo_piece is not None:
            if white:
                if promo_piece in self.white_captured:
                    self.white_captured.remove(promo_piece)
            else:
                if promo_piece in self.black_captured:
                    self.black_captured.remove(promo_piece)

        self.board[fsq] = 0
        self.board[tsq] = np.int8(final)
        self.hash      ^= ZOBRIST[final, tsq]
        if final in (1, 6): self.pawn_hash ^= Z_PAWN[final, tsq]
        self.material  += sign * (int(MAT[final]) + int(PST[final, tsq]))
        self.hash      ^= Z_TURN

        return moved, captured, promo_piece, old_halfmove

    def unmake(self, mv: int, moved, captured, promo_piece, old_halfmove):
        sr, sc, dr, dc = dec(mv)
        fsq   = sr*6+sc
        tsq   = dr*6+dc
        final = int(self.board[tsq])
        white = moved in WHITE_PIECES

        self.halfmove_clock = old_halfmove

        self.hash ^= Z_TURN
        self.hash ^= ZOBRIST[final, tsq]
        if final in (1, 6): self.pawn_hash ^= Z_PAWN[final, tsq]
        sign = 1 if white else -1
        self.material -= sign * (int(MAT[final]) + int(PST[final, tsq]))

        if promo_piece is not None:
            if white:
                self.white_captured.append(promo_piece)
            else:
                self.black_captured.append(promo_piece)

        self.board[fsq] = np.int8(moved)
        self.hash ^= ZOBRIST[moved, fsq]
        if moved in (1, 6): self.pawn_hash ^= Z_PAWN[moved, fsq]
        self.material += sign * (int(MAT[moved]) + int(PST[moved, fsq]))

        if captured:
            self.board[tsq] = np.int8(captured)
            self.hash ^= ZOBRIST[captured, tsq]
            if captured in (1, 6): self.pawn_hash ^= Z_PAWN[captured, tsq]
            sc2 = 1 if captured in WHITE_PIECES else -1
            self.material += sc2 * (int(MAT[captured]) + int(PST[captured, tsq]))
            if captured in WHITE_PIECES:
                if captured in self.white_captured:
                    self.white_captured.remove(captured)
            else:
                if captured in self.black_captured:
                    self.black_captured.remove(captured)
        else:
            self.board[tsq] = 0

    def find_king(self, white: bool) -> int:
        t = WHITE_KING if white else BLACK_KING
        h = np.where(self.board == t)[0]
        return int(h[0]) if len(h) else -1

    def attacked(self, sq: int, by_white: bool) -> bool:
        kr, kc = sq // 6, sq % 6
        ep = WHITE_PAWN   if by_white else BLACK_PAWN
        en = WHITE_KNIGHT if by_white else BLACK_KNIGHT
        eb = WHITE_BISHOP if by_white else BLACK_BISHOP
        eq = WHITE_QUEEN  if by_white else BLACK_QUEEN
        ek = WHITE_KING   if by_white else BLACK_KING

        pd = -1 if by_white else 1
        for dc in (-1, 1):
            r, c = kr+pd, kc+dc
            if 0<=r<6 and 0<=c<6 and self.board[r*6+c] == ep: return True

        for dr, dc in ((2,1),(2,-1),(-2,1),(-2,-1),(1,2),(1,-2),(-1,2),(-1,-2)):
            r, c = kr+dr, kc+dc
            if 0<=r<6 and 0<=c<6 and self.board[r*6+c] == en: return True

        for dr, dc in ((-1,-1),(-1,1),(1,-1),(1,1),(-1,0),(1,0),(0,-1),(0,1)):
            r, c = kr+dr, kc+dc
            while 0<=r<6 and 0<=c<6:
                p = int(self.board[r*6+c])
                if p:
                    diag = (dr != 0 and dc != 0)
                    if (diag and p in (eb, eq)) or (not diag and p == eq): return True
                    break
                r += dr; c += dc

        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0: continue
                r, c = kr+dr, kc+dc
                if 0<=r<6 and 0<=c<6 and self.board[r*6+c] == ek: return True
        return False

    def in_check(self, white: bool) -> bool:
        sq = self.find_king(white)
        return sq == -1 or self.attacked(sq, not white)

    def can_promote(self, white: bool) -> bool:
        return bool(self.white_captured if white else self.black_captured)

    def is_pawn_endgame(self) -> bool:
        for sq in range(NUM_SQUARES):
            p = int(self.board[sq])
            if p and p not in (WHITE_PAWN, BLACK_PAWN, WHITE_KING, BLACK_KING):
                return False
        return True

    def game_phase(self) -> float:
        mat = 0
        for sq in range(NUM_SQUARES):
            p = int(self.board[sq])
            if p and p not in (WHITE_PAWN, BLACK_PAWN, WHITE_KING, BLACK_KING):
                mat += int(MAT[p])
        return min(mat, PHASE_TOTAL) / PHASE_TOTAL

    def decay_history(self):
        self.history = (self.history * 3 // 4).astype(np.int32)



def _ray_clear(board, fr_sq: int, to_sq: int) -> bool:
    fr, fc = fr_sq // 6, fr_sq % 6
    tr, tc = to_sq // 6, to_sq % 6
    dr = 0 if tr == fr else (1 if tr > fr else -1)
    dc = 0 if tc == fc else (1 if tc > fc else -1)
    r, c = fr + dr, fc + dc
    while (r, c) != (tr, tc):
        if board[r*6 + c] != 0:
            return False
        r += dr; c += dc
    return True

def see(bs: BoardState, to_sq: int, target_val: int,
        fr_sq: int, att_val: int, white: bool) -> int:
    gain    = [0] * 16
    gain[0] = target_val
    used    = set()
    used.add(fr_sq)
    d       = 1

    board = bs.board
    while True:
        gain[d] = att_val - gain[d-1]
        best_val = 99999; best_sq = -1
        side = (not white) if (d % 2 == 1) else white
        own  = WHITE_PIECES if side else BLACK_PIECES

        for sq in range(NUM_SQUARES):
            if sq in used: continue
            p = int(board[sq])
            if p not in own: continue
            pv = int(MAT[p])
            if pv >= best_val: continue
            r, c   = sq // 6, sq % 6
            tr, tc = to_sq // 6, to_sq % 6
            dr, dc = tr - r, tc - c
            reachable = False
            if p in (WHITE_PAWN, BLACK_PAWN):
                pd = 1 if p == WHITE_PAWN else -1
                reachable = (dr == pd and abs(dc) == 1)
            elif p in (WHITE_KNIGHT, BLACK_KNIGHT):
                reachable = (abs(dr), abs(dc)) in ((2,1),(1,2))
            elif p in (WHITE_BISHOP, BLACK_BISHOP):
                reachable = (abs(dr) == abs(dc) and dr != 0 and
                             _ray_clear(board, sq, to_sq))
            elif p in (WHITE_QUEEN, BLACK_QUEEN):
                on_diag  = (abs(dr) == abs(dc) and dr != 0)
                on_ortho = (dr == 0 or dc == 0) and not (dr == 0 and dc == 0)
                reachable = ((on_diag or on_ortho) and
                             _ray_clear(board, sq, to_sq))
            elif p in (WHITE_KING, BLACK_KING):
                reachable = (max(abs(dr), abs(dc)) == 1)
            if reachable:
                best_val = pv; best_sq = sq

        if best_sq == -1: break
        used.add(best_sq)
        att_val = best_val
        d += 1
        if d >= 16: break

    while d > 1:
        d -= 1
        gain[d-1] = -max(-gain[d-1], gain[d])
    return gain[0]



def gen_captures(bs: BoardState, white: bool) -> list:
    moves = []
    own   = WHITE_PIECES if white else BLACK_PIECES
    enemy = BLACK_PIECES if white else WHITE_PIECES
    board = bs.board
    for sq in range(NUM_SQUARES):
        p = int(board[sq])
        if p == 0 or p not in own: continue
        r, c = sq // 6, sq % 6
        _add_captures(board, r, c, p, own, enemy, moves)
    return moves

def gen_all(bs: BoardState, white: bool) -> list:
    moves = []
    own   = WHITE_PIECES if white else BLACK_PIECES
    enemy = BLACK_PIECES if white else WHITE_PIECES
    board = bs.board
    for sq in range(NUM_SQUARES):
        p = int(board[sq])
        if p == 0 or p not in own: continue
        r, c = sq // 6, sq % 6
        _add_all(bs, board, r, c, p, own, enemy, moves, white)
    return moves

def _add_captures(board, r, c, p, own, enemy, out):
    if p in (WHITE_PAWN, BLACK_PAWN):
        d = 1 if p == WHITE_PAWN else -1
        for dc in (-1, 1):
            nr, nc = r+d, c+dc
            if 0<=nr<6 and 0<=nc<6:
                t = int(board[nr*6+nc])
                if t in enemy: out.append(enc(r, c, nr, nc))
    elif p in (WHITE_KNIGHT, BLACK_KNIGHT):
        for dr, dc in ((2,1),(2,-1),(-2,1),(-2,-1),(1,2),(1,-2),(-1,2),(-1,-2)):
            nr, nc = r+dr, c+dc
            if 0<=nr<6 and 0<=nc<6 and int(board[nr*6+nc]) in enemy:
                out.append(enc(r, c, nr, nc))
    elif p in (WHITE_BISHOP, BLACK_BISHOP):
        for dr, dc in ((-1,-1),(-1,1),(1,-1),(1,1)):
            nr, nc = r+dr, c+dc
            while 0<=nr<6 and 0<=nc<6:
                t = int(board[nr*6+nc])
                if t:
                    if t in enemy: out.append(enc(r, c, nr, nc))
                    break
                nr += dr; nc += dc
    elif p in (WHITE_QUEEN, BLACK_QUEEN):
        for dr, dc in ((-1,-1),(-1,1),(1,-1),(1,1),(-1,0),(1,0),(0,-1),(0,1)):
            nr, nc = r+dr, c+dc
            while 0<=nr<6 and 0<=nc<6:
                t = int(board[nr*6+nc])
                if t:
                    if t in enemy: out.append(enc(r, c, nr, nc))
                    break
                nr += dr; nc += dc
    elif p in (WHITE_KING, BLACK_KING):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0: continue
                nr, nc = r+dr, c+dc
                if 0<=nr<6 and 0<=nc<6 and int(board[nr*6+nc]) in enemy:
                    out.append(enc(r, c, nr, nc))

def _add_all(bs, board, r, c, p, own, enemy, out, white):
    if p in (WHITE_PAWN, BLACK_PAWN):
        d   = 1 if p == WHITE_PAWN else -1
        nr  = r + d
        if 0 <= nr < 6:
            is_promo_rank = (p == WHITE_PAWN and nr == 5) or \
                            (p == BLACK_PAWN  and nr == 0)
            if not (is_promo_rank and not bs.can_promote(white)):
                if board[nr*6 + c] == 0:
                    out.append(enc(r, c, nr, c))
            for dc in (-1, 1):
                nc = c + dc
                if 0 <= nc < 6:
                    t = int(board[nr*6 + nc])
                    if t in enemy:
                        if not (is_promo_rank and not bs.can_promote(white)):
                            out.append(enc(r, c, nr, nc))
    elif p in (WHITE_KNIGHT, BLACK_KNIGHT):
        for dr, dc in ((2,1),(2,-1),(-2,1),(-2,-1),(1,2),(1,-2),(-1,2),(-1,-2)):
            nr, nc = r+dr, c+dc
            if 0<=nr<6 and 0<=nc<6:
                t = int(board[nr*6+nc])
                if t == 0 or t in enemy: out.append(enc(r, c, nr, nc))
    elif p in (WHITE_BISHOP, BLACK_BISHOP):
        for dr, dc in ((-1,-1),(-1,1),(1,-1),(1,1)):
            nr, nc = r+dr, c+dc
            while 0<=nr<6 and 0<=nc<6:
                t = int(board[nr*6+nc])
                if t == 0: out.append(enc(r, c, nr, nc))
                else:
                    if t in enemy: out.append(enc(r, c, nr, nc))
                    break
                nr += dr; nc += dc
    elif p in (WHITE_QUEEN, BLACK_QUEEN):
        for dr, dc in ((-1,-1),(-1,1),(1,-1),(1,1),(-1,0),(1,0),(0,-1),(0,1)):
            nr, nc = r+dr, c+dc
            while 0<=nr<6 and 0<=nc<6:
                t = int(board[nr*6+nc])
                if t == 0: out.append(enc(r, c, nr, nc))
                else:
                    if t in enemy: out.append(enc(r, c, nr, nc))
                    break
                nr += dr; nc += dc
    elif p in (WHITE_KING, BLACK_KING):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0: continue
                nr, nc = r+dr, c+dc
                if 0<=nr<6 and 0<=nc<6:
                    t = int(board[nr*6+nc])
                    if t == 0 or t in enemy:
                        out.append(enc(r, c, nr, nc))



def score_move(mv: int, bs: BoardState, ply: int, tt_mv: int,
               prev_mv: int, white: bool) -> int:
    if mv == tt_mv and tt_mv != 0:
        return 3_000_000

    sr, sc, dr, dc = dec(mv)
    to_sq  = dr*6+dc
    fr_sq  = sr*6+sc
    victim = int(bs.board[to_sq])
    att    = int(bs.board[fr_sq])

    if victim:
        see_val = see(bs, to_sq, int(MAT[victim]), fr_sq, int(MAT[att]), white)
        return 2_000_000 + see_val

    if bs.killers[ply][0] == mv: return 1_000_000
    if bs.killers[ply][1] == mv:   return 900_000

    if prev_mv > 0:
        psr, psc, pdr, pdc = dec(prev_mv)
        if int(bs.counter[psr*6+psc, pdr*6+pdc]) == mv:
            return 800_000

    return int(bs.history[att, to_sq])

def order_moves(moves: list, bs: BoardState, ply: int,
                tt_mv: int, prev_mv: int, white: bool) -> list:
    return sorted(moves,
                  key=lambda m: score_move(m, bs, ply, tt_mv, prev_mv, white),
                  reverse=True)



def eval_pawns(bs: BoardState) -> int:
    cached = pawn_hash_lookup(bs.pawn_hash)
    if cached is not None:
        return cached

    score = 0
    board = bs.board

    w_files = [0]*6
    b_files = [0]*6
    for sq in range(NUM_SQUARES):
        p = int(board[sq])
        if p == WHITE_PAWN: w_files[sq % 6] += 1
        if p == BLACK_PAWN: b_files[sq % 6] += 1

    for f in range(6):
        if w_files[f] > 1: score -= 20 * (w_files[f] - 1)
        if b_files[f] > 1: score += 20 * (b_files[f] - 1)

    for f in range(6):
        left  = (f > 0)
        right = (f < 5)
        if w_files[f]:
            if (not left or not w_files[f-1]) and (not right or not w_files[f+1]):
                score -= 15
        if b_files[f]:
            if (not left or not b_files[f-1]) and (not right or not b_files[f+1]):
                score += 15

    for sq in range(NUM_SQUARES):
        p = int(board[sq])
        r, f = sq // 6, sq % 6

        if p == WHITE_PAWN:
            is_passed = True
            for bf in range(max(0, f-1), min(6, f+2)):
                for br in range(r+1, 6):
                    if board[br*6 + bf] == BLACK_PAWN:
                        is_passed = False; break
                if not is_passed: break
            if is_passed:
                score += 30 + r * 12

        elif p == BLACK_PAWN:
            is_passed = True
            for bf in range(max(0, f-1), min(6, f+2)):
                for br in range(0, r):
                    if board[br*6 + bf] == WHITE_PAWN:
                        is_passed = False; break
                if not is_passed: break
            if is_passed:
                score -= 30 + (5-r) * 12

    pawn_hash_store(bs.pawn_hash, score)
    return score


def eval_king_safety(bs: BoardState) -> int:
    score = 0
    board = bs.board

    for white_king in (True, False):
        king_sq = bs.find_king(white_king)
        if king_sq == -1: continue
        kr, kc = king_sq // 6, king_sq % 6
        sign   = 1 if white_king else -1

        attack_penalty = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = kr+dr, kc+dc
                if 0<=nr<6 and 0<=nc<6:
                    if bs.attacked(nr*6+nc, not white_king):
                        attack_penalty += 12

        shelter_bonus = 0
        pawn = WHITE_PAWN if white_king else BLACK_PAWN
        for f in range(max(0, kc-1), min(6, kc+2)):
            fwd = 1 if white_king else -1
            for step in (1, 2):
                r2 = kr + fwd * step
                if 0<=r2<6 and board[r2*6+f] == pawn:
                    shelter_bonus += 10
                    break

        score += sign * (shelter_bonus - attack_penalty)

    return score


def eval_endgame_kings(bs: BoardState) -> int:
    board = bs.board
    score = 0

    wk_sq = bs.find_king(white=True)
    bk_sq = bs.find_king(white=False)
    if wk_sq == -1 or bk_sq == -1:
        return 0

    wkr, wkc = wk_sq // 6, wk_sq % 6
    bkr, bkc = bk_sq // 6, bk_sq % 6

    king_dist = max(abs(wkr - bkr), abs(wkc - bkc))

    w_mat = sum(int(MAT[int(board[sq])]) for sq in range(NUM_SQUARES)
                if board[sq] in WHITE_PIECES and board[sq] != WHITE_KING)
    b_mat = sum(int(MAT[int(board[sq])]) for sq in range(NUM_SQUARES)
                if board[sq] in BLACK_PIECES and board[sq] != BLACK_KING)

    if w_mat > b_mat:
        score += (6 - king_dist) * 5
    elif b_mat > w_mat:
        score -= (6 - king_dist) * 5

    for sq in range(NUM_SQUARES):
        p = int(board[sq])
        r, f = sq // 6, sq % 6

        if p == WHITE_PAWN:
            wk_pawn_dist = max(abs(wkr - r), abs(wkc - f))
            bk_pawn_dist = max(abs(bkr - r), abs(bkc - f))
            score += (6 - wk_pawn_dist) * 2
            score -= (6 - bk_pawn_dist) * 2

        elif p == BLACK_PAWN:
            wk_pawn_dist = max(abs(wkr - r), abs(wkc - f))
            bk_pawn_dist = max(abs(bkr - r), abs(bkc - f))
            score -= (6 - bk_pawn_dist) * 2
            score += (6 - wk_pawn_dist) * 2

    same_file = (wkc == bkc)
    same_rank = (wkr == bkr)
    if same_file and abs(wkr - bkr) == 2:
        score += 15
    elif same_rank and abs(wkc - bkc) == 2:
        score += 15

    return score


def evaluate(bs: BoardState, white: bool) -> int:
    phase = bs.game_phase()

    mg_score = 0
    eg_score = 0
    for sq in range(NUM_SQUARES):
        p = int(bs.board[sq])
        if p == 0: continue
        s = 1 if p in WHITE_PIECES else -1
        mg_score += s * (int(MAT[p]) + int(PST_MG[p, sq]))
        eg_score += s * (int(MAT[p]) + int(PST_EG[p, sq]))

    tapered = int(phase * mg_score + (1.0 - phase) * eg_score)

    pawn_score = eval_pawns(bs)
    ks_score   = int(eval_king_safety(bs) * phase)

    w_bish = int(np.sum(bs.board == WHITE_BISHOP))
    b_bish = int(np.sum(bs.board == BLACK_BISHOP))
    bishop_pair = (40 if w_bish >= 2 else 0) - (40 if b_bish >= 2 else 0)

    w_mob = b_mob = 0
    for sq in range(NUM_SQUARES):
        p = int(bs.board[sq])
        if p == 0: continue
        r, c = sq // 6, sq % 6
        if p in WHITE_PIECES:
            w_mob += _piece_mobility(bs.board, r, c, p, WHITE_PIECES, BLACK_PIECES)
        else:
            b_mob += _piece_mobility(bs.board, r, c, p, BLACK_PIECES, WHITE_PIECES)
    mob_score = (w_mob - b_mob) * 3

    eg_king_score = int(eval_endgame_kings(bs) * (1.0 - phase))

    score = tapered + pawn_score + ks_score + bishop_pair + mob_score + eg_king_score
    return score if white else -score

def _piece_mobility(board, r, c, p, own, enemy):
    cnt = 0
    if p in (WHITE_PAWN, BLACK_PAWN):
        d = 1 if p == WHITE_PAWN else -1
        if 0<=r+d<6:
            cnt += board[(r+d)*6+c] == 0
            for dc in (-1, 1):
                nc = c+dc
                if 0<=nc<6 and int(board[(r+d)*6+nc]) in enemy: cnt += 1
    elif p in (WHITE_KNIGHT, BLACK_KNIGHT):
        for dr, dc in ((2,1),(2,-1),(-2,1),(-2,-1),(1,2),(1,-2),(-1,2),(-1,-2)):
            nr, nc = r+dr, c+dc
            if 0<=nr<6 and 0<=nc<6:
                t = int(board[nr*6+nc])
                if t == 0 or t in enemy: cnt += 1
    else:
        dirs = ((-1,-1),(-1,1),(1,-1),(1,1)) if p in (3, 8) else \
               ((-1,-1),(-1,1),(1,-1),(1,1),(-1,0),(1,0),(0,-1),(0,1))
        for dr, dc in dirs:
            nr, nc = r+dr, c+dc
            while 0<=nr<6 and 0<=nc<6:
                t = int(board[nr*6+nc])
                cnt += 1
                if t: break
                nr += dr; nc += dc
    return cnt



DELTA = 200

def quiescence(bs: BoardState, alpha: int, beta: int,
               white: bool, ply: int) -> int:

    if _tick(): return evaluate(bs, white)   # time-expired fast path

    if bs.find_king(white) == -1:
        return -MATE + ply

    stand = evaluate(bs, white)
    if stand >= beta: return beta
    if stand > alpha: alpha = stand

    caps = gen_captures(bs, white)
    caps.sort(key=lambda m: see(bs,
                                dec(m)[2]*6+dec(m)[3],
                                int(MAT[int(bs.board[dec(m)[2]*6+dec(m)[3]])]),
                                dec(m)[0]*6+dec(m)[1],
                                int(MAT[int(bs.board[dec(m)[0]*6+dec(m)[1]])]),
                                white),
              reverse=True)

    for mv in caps:
        sr, sc, dr, dc = dec(mv)
        to_sq  = dr*6+dc
        victim = int(bs.board[to_sq])

        if stand + int(MAT[victim]) + DELTA < alpha:
            continue

        moved, captured, promo, old_hm = bs.make(mv)
        ksq = bs.find_king(white)
        if ksq == -1 or bs.attacked(ksq, not white):
            bs.unmake(mv, moved, captured, promo, old_hm); continue

        score = -quiescence(bs, -beta, -alpha, not white, ply+1)
        bs.unmake(mv, moved, captured, promo, old_hm)

        if score >= beta: return beta
        if score > alpha: alpha = score

    return alpha



MATE      = 19000
NULL_R    = 2
LMR_MIN   = 4
LMR_DEPTH = 3

FUTILITY       = [0, 100, 200, 350, 500]
RAZOR_MARGIN   = 300
FUTILITY_MAX_DEPTH = 3

LMP_COUNTS   = [0, 8, 14, 22, 36]

PROBCUT_MARGIN = 150

def negamax(bs: BoardState, depth: int, alpha: int, beta: int,
            white: bool, ply: int, prev_mv: int,
            null_ok: bool,
            pos_history: list) -> tuple:

    # ── Time check via node counter (no ply-modulo guessing) ────────
    if _tick():
        return evaluate(bs, white), 0

    orig_alpha = alpha

    # ── Repetition / 50-move draw ────────────────────────────────────
    cur_hash = int(bs.hash)
    if pos_history.count(cur_hash) >= 2:
        return 0, 0
    if bs.halfmove_clock >= 100:
        return 0, 0

    # ── Transposition table — full 64-bit lookup ─────────────────────
    h = bs.hash
    tt_score, tt_mv, tt_hit = tt_lookup(h, depth, alpha, beta)
    if tt_hit:
        return tt_score, tt_mv if tt_mv else 0

    in_check = bs.in_check(white)
    if in_check: depth += 1

    if depth <= 0:
        return quiescence(bs, alpha, beta, white, ply), 0

    static = evaluate(bs, white)

    # ── Razoring ─────────────────────────────────────────────────────
    if (not in_check and depth <= 2 and
            static + RAZOR_MARGIN * depth < alpha):
        q = quiescence(bs, alpha - 1, alpha, white, ply)
        if q < alpha:
            return q, 0

    # ── Null-move pruning ─────────────────────────────────────────────
    if (null_ok and not in_check and depth >= 3 and
            abs(alpha) < MATE and abs(beta) < MATE and
            static >= beta and not bs.is_pawn_endgame()):
        bs.hash ^= Z_TURN
        bs.halfmove_clock += 1
        pos_history.append(int(bs.hash))
        ns, _ = negamax(bs, depth-1-NULL_R, -beta, -beta+1,
                        not white, ply+1, NULL_MOVE, False, pos_history)
        ns = -ns
        pos_history.pop()
        bs.halfmove_clock -= 1
        bs.hash ^= Z_TURN
        if ns >= beta:
            return beta, 0

    # ── Internal iterative reduction (no TT move) ─────────────────────
    if depth >= 4 and tt_mv == 0:
        depth -= 1

    # ── ProbCut ───────────────────────────────────────────────────────
    if depth >= 5 and not in_check and abs(beta) < MATE:
        pc_beta  = beta + PROBCUT_MARGIN
        pc_depth = depth - 4
        pc_caps  = gen_captures(bs, white)
        pc_caps.sort(key=lambda m: int(MAT[int(bs.board[dec(m)[2]*6+dec(m)[3]])]),
                     reverse=True)
        for pc_mv in pc_caps[:4]:
            sr2, sc2, dr2, dc2 = dec(pc_mv)
            moved2, cap2, promo2, old_hm2 = bs.make(pc_mv)
            ksq2 = bs.find_king(white)
            if ksq2 == -1 or bs.attacked(ksq2, not white):
                bs.unmake(pc_mv, moved2, cap2, promo2, old_hm2)
                continue
            pos_history.append(int(bs.hash))
            pc_score, _ = negamax(bs, pc_depth, -pc_beta, -pc_beta+1,
                                  not white, ply+1, pc_mv, False, pos_history)
            pc_score = -pc_score
            pos_history.pop()
            bs.unmake(pc_mv, moved2, cap2, promo2, old_hm2)
            if pc_score >= pc_beta:
                return pc_beta, pc_mv

    moves = gen_all(bs, white)
    if not moves:
        return (-MATE+ply, 0) if in_check else (0, 0)

    ordered   = order_moves(moves, bs, ply, tt_mv or 0, prev_mv, white)
    best_val  = -10**7
    best_mv   = ordered[0]
    legal_cnt = 0
    quiet_cnt = 0

    for i, mv in enumerate(ordered):
        sr, sc, dr, dc = dec(mv)
        moved, captured, promo, old_hm = bs.make(mv)

        ksq = bs.find_king(white)
        if ksq == -1 or bs.attacked(ksq, not white):
            bs.unmake(mv, moved, captured, promo, old_hm); continue

        legal_cnt += 1
        is_cap    = (captured != 0)
        gives_chk = bs.in_check(not white)

        if not is_cap: quiet_cnt += 1

        # ── Late Move Pruning ─────────────────────────────────────────
        if (not in_check and not is_cap and not gives_chk and
                depth <= 4 and depth >= 1 and
                quiet_cnt > LMP_COUNTS[min(depth, 4)]):
            bs.unmake(mv, moved, captured, promo, old_hm)
            continue

        # ── Futility pruning ──────────────────────────────────────────
        if (not in_check and not is_cap and not gives_chk and
                depth <= FUTILITY_MAX_DEPTH and legal_cnt > 1):
            fut_margin = FUTILITY[min(depth, 4)]
            if static + fut_margin <= alpha:
                bs.unmake(mv, moved, captured, promo, old_hm); continue

        # ── Late Move Reduction ───────────────────────────────────────
        reduce = 0
        if (i >= LMR_MIN and depth >= LMR_DEPTH and
                not is_cap and not gives_chk and not in_check):
            reduce = 1 + (i >= LMR_MIN * 2)

        pos_history.append(int(bs.hash))

        if i == 0:
            score, _ = negamax(bs, depth-1, -beta, -alpha,
                               not white, ply+1, mv, True, pos_history)
            score = -score
        else:
            score, _ = negamax(bs, depth-1-reduce, -alpha-1, -alpha,
                               not white, ply+1, mv, True, pos_history)
            score = -score
            if score > alpha and (score < beta or reduce > 0):
                score, _ = negamax(bs, depth-1, -beta, -alpha,
                                   not white, ply+1, mv, True, pos_history)
                score = -score

        pos_history.pop()
        bs.unmake(mv, moved, captured, promo, old_hm)

        if score > best_val:
            best_val = score; best_mv = mv

        if score > alpha: alpha = score

        if alpha >= beta:
            if not is_cap:
                k = bs.killers[ply]
                if k[0] != mv: k[1] = k[0]; k[0] = mv
                bs.history[moved, dr*6+dc] += depth * depth
                if prev_mv > 0:
                    psr, psc, pdr, pdc = dec(prev_mv)
                    bs.counter[psr*6+psc, pdr*6+pdc] = np.int16(mv & 0x7FFF)
            break

    if legal_cnt == 0:
        return (-MATE+ply, 0) if in_check else (0, 0)

    flag = (TT_EXACT if orig_alpha < best_val < beta
            else TT_LOWER if best_val >= beta else TT_UPPER)
    tt_store(h, depth, best_val, flag, best_mv)

    return best_val, best_mv



ASP_DELTA = 25
ASP_WIDEN = 4

def iterative_deepening(bs: BoardState, white: bool,
                         max_depth: int = 14,
                         time_limit: float = 10.0,
                         root_pos_history: list = None) -> tuple:
    best_mv    = 0
    best_score = 0
    deadline   = time.time() + time_limit
    prev_score = 0

    if root_pos_history is None:
        root_pos_history = []

    pos_history = list(root_pos_history) + [int(bs.hash)]

    for depth in range(1, max_depth + 1):
        # ── Pre-depth time guard ──────────────────────────────────────
        if time.time() >= deadline:
            break

        # Reset node counter for each depth iteration.
        # This ensures the time flag is cleared between iterations so
        # a completed shallow search never blocks the next one.
        _reset_time_control(deadline)

        bs.decay_history()

        if depth <= 3:
            alpha, beta = -10**7, 10**7
        else:
            alpha = prev_score - ASP_DELTA
            beta  = prev_score + ASP_DELTA

        asp_delta = ASP_DELTA
        while True:
            score, mv = negamax(bs, depth, alpha, beta, white, 0, 0,
                                True, pos_history)

            # If time expired mid-search, trust the last *completed* result
            if _time_expired[0]:
                break

            if score <= alpha:
                alpha    -= asp_delta * ASP_WIDEN
                asp_delta *= ASP_WIDEN
            elif score >= beta:
                beta     += asp_delta * ASP_WIDEN
                asp_delta *= ASP_WIDEN
            else:
                break

            if alpha < -9000: alpha = -10**7
            if beta  >  9000: beta  =  10**7
            if alpha <= -10**7 and beta >= 10**7: break

        # Only commit best move if this depth finished cleanly
        if not _time_expired[0]:
            if mv and mv != 0: best_mv = mv
            best_score = score
            prev_score = score

        elapsed = time.time() - (deadline - time_limit)

        if best_mv:
            sr, sc, dr, dc = dec(best_mv)
            p = int(bs.board[sr*6+sc])
            mv_str = (f"{p}:{COL_TO_FILE[sc]}{sr+1}->"
                      f"{COL_TO_FILE[dc]}{dr+1}") if p else "---"
        else:
            mv_str = "---"

        status = "PARTIAL" if _time_expired[0] else "ok"
        print(f"  depth={depth:2d}  score={best_score:+6d}  "
              f"move={mv_str}  ({elapsed:.2f}s)  [{status}]")

        if abs(best_score) >= MATE - 50:
            print(f"  *** Forced mate in ~{(MATE - abs(best_score))//2+1} ***")
            break

        if _time_expired[0] or time.time() >= deadline - 0.05:
            break

    return best_mv, best_score



def _fischer_random_back_rank(white: bool) -> list:
    knight = WHITE_KNIGHT if white else BLACK_KNIGHT
    bishop = WHITE_BISHOP if white else BLACK_BISHOP
    queen  = WHITE_QUEEN  if white else BLACK_QUEEN
    king   = WHITE_KING   if white else BLACK_KING

    while True:
        positions  = list(range(6))
        even_files = [p for p in positions if p % 2 == 0]
        odd_files  = [p for p in positions if p % 2 == 1]
        if not even_files or not odd_files: continue

        b1 = random.choice(even_files)
        b2 = random.choice(odd_files)
        remaining = [p for p in positions if p != b1 and p != b2]
        random.shuffle(remaining)

        rank      = [0] * 6
        rank[b1]  = bishop
        rank[b2]  = bishop
        rank[remaining[0]] = king
        rank[remaining[1]] = queen
        rank[remaining[2]] = knight
        rank[remaining[3]] = knight
        return rank

def starting_board() -> np.ndarray:
    b = np.zeros(36, dtype=np.int8)
    white_back = _fischer_random_back_rank(white=True)
    black_back = _fischer_random_back_rank(white=False)
    for c, p in enumerate(white_back): b[c]      = p
    b[6:12]  = WHITE_PAWN
    b[24:30] = BLACK_PAWN
    for c, p in enumerate(black_back): b[30+c]   = p
    return b

def fixed_starting_board() -> np.ndarray:
    b = np.zeros(36, dtype=np.int8)
    for c, p in enumerate([2,3,4,5,3,2]): b[c]  = p
    b[6:12]  = 1
    b[24:30] = 6
    for c, p in enumerate([7,8,9,10,8,7]): b[30+c] = p
    return b



def _board_key(board: np.ndarray) -> tuple:
    return tuple(int(x) for x in board)

def _build_opening_book() -> dict:
    book = {}
    b    = fixed_starting_board()
    k0   = _board_key(b)
    book[k0] = [
        f"{WHITE_PAWN}:C2->C4",
        f"{WHITE_PAWN}:D2->D4",
        f"{WHITE_PAWN}:C2->C3",
        f"{WHITE_KNIGHT}:B1->C3",
        f"{WHITE_KNIGHT}:E1->D3",
    ]
    return book

OPENING_BOOK = _build_opening_book()

def book_lookup(board: np.ndarray, white: bool) -> Optional[str]:
    key = _board_key(board)
    if key in OPENING_BOOK:
        candidates = OPENING_BOOK[key]
        if white:
            return random.choice(candidates)
    return None



def print_board(b):
    sym = {0:'.',1:'P',2:'N',3:'B',4:'Q',5:'K',
                  6:'p',7:'n',8:'b',9:'q',10:'k'}
    g = b.reshape(6,6) if b.ndim == 1 else b
    print("  A B C D E F")
    for r in range(5, -1, -1):
        print(f"{r+1} {' '.join(sym[int(g[r,c])] for c in range(6))} {r+1}")
    print("  A B C D E F\n")

def move_str(mv: int, board: np.ndarray) -> str:
    sr, sc, dr, dc = dec(mv)
    p = int(board[sr*6+sc])
    return f"{p}:{COL_TO_FILE[sc]}{sr+1}->{COL_TO_FILE[dc]}{dr+1}"



def get_best_move(board: np.ndarray,
                  playing_white:  bool  = True,
                  max_depth:      int   = 14,
                  time_limit:     float = 10.0,
                  white_captured: list  = None,
                  black_captured: list  = None,
                  halfmove_clock: int   = 0,
                  pos_history:    list  = None) -> Optional[str]:

    flat = board.flatten().astype(np.int8)

    book_mv = book_lookup(flat, playing_white)
    if book_mv is not None:
        print(f"\n[MAX Engine v6] Opening book: {book_mv}")
        return book_mv

    bs = BoardState(flat, white_captured, black_captured, halfmove_clock)

    print(f"\n[MAX Engine v6] {'White' if playing_white else 'Black'} | "
          f"depth≤{max_depth} | {time_limit}s | "
          f"hm_clock={halfmove_clock}")

    mv, score = iterative_deepening(bs, playing_white, max_depth, time_limit,
                                    pos_history or [])
    if not mv: return None

    sr, sc, dr, dc = dec(mv)
    p = int(flat[sr*6+sc])

    move = f"{p}:{COL_TO_FILE[sc]}{sr+1}->{COL_TO_FILE[dc]}{dr+1}"

    # Always print promotion format
    if p == 1 and dr == 5:
        promo = 4
    elif p == 6 and dr == 0:
        promo = 9
    else:
        promo = p   # if not promotion just repeat same piece

    move += f"={promo}"

    return move


if __name__ == '__main__':
    print("RoboGambit MAX Engine v6.0\n")

    tt_bytes = (TT_HASH.nbytes + TT_SCORE.nbytes +
                TT_DEPTH.nbytes + TT_FLAG.nbytes + TT_MOVE.nbytes)
    tt_mb  = tt_bytes / 1024**2
    tt_gb  = tt_bytes / 1e9          # decimal GB (what OS reports)
    print(f"TT memory used  : {tt_mb:,.0f} MiB  =  {tt_gb:.2f} GB  (decimal)")
    print(f"TT entries      : {int(TT_SIZE):,}  (2^29)")
    print(f"Hash width      : 64-bit  |  Score dtype : int16 (safe for ±19000)\n")

    board = starting_board()
    print("Fischer Random starting position:")
    print_board(board)

    white           = True
    white_captured  = []
    black_captured  = []
    halfmove_clock  = 0
    pos_history     = []

    for move_num in range(1, 40):
        result = get_best_move(
            board,
            playing_white  = white,
            max_depth      = 14,
            time_limit     = 10.0,
            white_captured = white_captured,
            black_captured = black_captured,
            halfmove_clock = halfmove_clock,
            pos_history    = pos_history,
        )

        if result is None:
            print("No legal moves — game over.")
            break

        print(f"\nMove {move_num} ({'White' if white else 'Black'}): {result}\n")

        bs = BoardState(board.flatten().astype(np.int8),
                        white_captured = list(white_captured),
                        black_captured = list(black_captured),
                        halfmove_clock = halfmove_clock)

        all_mv  = gen_all(bs, white)
        applied = False
        for mv in all_mv:
            if move_str(mv, board.flatten()) == result:
                moved, captured, promo, old_hm = bs.make(mv)
                white_captured = list(bs.white_captured)
                black_captured = list(bs.black_captured)
                halfmove_clock = bs.halfmove_clock
                board = bs.board.copy()
                pos_history.append(int(bs.hash))
                applied = True
                break

        if not applied:
            print(f"  [WARN] Could not find move {result} in legal moves.")
            break

        print_board(board)

        if halfmove_clock >= 100:
            print("Draw by 50-move rule.")
            break

        white = not white
