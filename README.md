# pyfastchess

`pyfastchess` is a Python package that provides **fast chess move generation and board state utilities** via a C++ backend with [pybind11](https://github.com/pybind/pybind11).  

It is designed as a drop-in replacement for slower pure-Python libraries (like [`python-chess`](https://github.com/niklasf/python-chess)) when you need blazing-fast move generation or efficient tensor encodings for machine learning.

---

## 🚀 Features
- Wraps [Disservin’s `chess-library`](https://github.com/Disservin/chess-library) (C++17) for high-performance board and move operations.  
- Exposes a Pythonic API for:
  - Creating boards from FEN
  - Generating legal moves
  - Pushing/popping moves
  - Accessing piece locations and bitboards
  - Exporting tensor encodings (e.g. `planes_signed()`) for neural nets  
- Built with [pybind11](https://github.com/pybind/pybind11) and [scikit-build-core](https://scikit-build-core.readthedocs.io).

---

## 📦 Installation
Clone this repo with submodules (to pull in the chess-library dependency):

```bash
git clone --recursive https://github.com/yourname/pyfastchess.git
cd pyfastchess
pip install -e .
```

If you already cloned without `--recursive`, run:

```bash
git submodule update --init --recursive
```

---

## 🧑‍💻 Usage

```python
import pyfastchess as fc

# Start a new game
b = fc.Board()

print("FEN:", b.fen())
print("Legal moves:", b.legal_moves()[:10])

# push UCI moves e.g. 1. E4 E5
b.push_uci("e2e4")
b.push_uci("e7e5")

# show move history
print("move history:", b.history_uci())

# retrieve fen
print('FEN after E4 E5:', b.fen())

# detect captures
b.push_uci('d2d4')
print('is exd5 a capture?', b.is_capture('e5d4'))

# get SAN notation
move_uci = 'e5d4'
print(f"UCI {move_uci}, SAN {b.san(move_uci)}")

# unmake moves
b.unmake()
print("history after unmaking d2d4:", b.history_uci())

# show how many pieces are on the board (for e.g. endgame detection)
print(f"{b.piece_count()} pieces on the board")
b = fc.Board()
for move in ['e2e4', 'd7d5', 'e4d5']:
    b.push_uci(move)

print(f"{b.piece_count()} pieces on the board after exd5")

# query for pieces at squares
print("piece on square 0 (A1):", b.piece_at(0)) # 4 = white rook
print("piece on square 63 (H8):", b.piece_at(63))# -4 = black rook
print("piece on square 24 (A4):", b.piece_at(24))# 0 = empty

# get color x square attacker information
b = fc.Board()
for move in ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'd2d4']:
    b.push_uci(move)

# piece lookup
pl = {
    0: None, 1: 'Pawn', 2: 'Knight', 3: 'Bishop',
    4: 'Rook', 5: 'Queen', 6: 'King'
}

# the E5 square is square number 36

on_e5 = b.piece_type_at(36)
# could also use b.piece_at() but that gives negatives for b

# color is b for black, w for white
clr = 'Black' if b.piece_color_at(36) == 'b' else 'White'
print(f"There is a {clr} {pl[on_e5]} on E5.")

# from which squares does white attack e5?
w_attacks_e5 = b.attackers_list('w', 36)

# with which pieces does white attack e5?
white_pcs = [b.piece_type_at(s) for s in w_attacks_e5]
print(f"White attacks {pl[on_e5]} on e5 with {[pl[p] for p in white_pcs]}")

b_attacks_e5 = b.attackers_list('b', 36)
black_pcs = [b.piece_type_at(s) for s in b_attacks_e5]
print(f"Black defends {pl[on_e5]} on e5 with {[pl[p] for p in black_pcs]}")


# create np arrays for tensorflow
# represents the board as 14x64,
# the last 5 board states are 'stacked' to encode history 
planes = b.stacked_planes(5)
print(planes.shape)   # (8, 8, 70)

```

---

## 📜 License

- **pyfastchess bindings (this repo):** MIT License © 2025 bagtf3
- **chess-library (submodule):** MIT License © 2021–present Disservin.  
- The `chess-library` code is included here as a **git submodule**. You can find its source and license at [github.com/Disservin/chess-library](https://github.com/Disservin/chess-library).  

---

## 🙏 Credits

- [Disservin’s chess-library](https://github.com/Disservin/chess-library) — high-performance C++ chess core that powers this package.  
- [pybind11](https://github.com/pybind/pybind11) — modern C++ ↔ Python bindings.  
- [scikit-build-core](https://scikit-build-core.readthedocs.io) — Python build system.  
