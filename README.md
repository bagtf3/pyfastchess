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

b.push_uci("e2e4")
b.push_uci("e7e5")

planes = b.planes_signed(stm_pov=True)
print(planes.shape)   # (6, 8, 8)
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
