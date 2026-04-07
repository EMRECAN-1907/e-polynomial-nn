# e-Polynomial Input Compression for Neural Networks

A simple yet effective input compression method for neural networks using e-polynomial transformation.

## What is this?

Instead of feeding raw high-dimensional inputs into a neural network, this method compresses them using a polynomial evaluated at Euler's number (e ≈ 2.718):

```
[a, b, c, d] → a·e³ + b·e² + c·e + d = single scalar
```

Because e is transcendental, this transformation is **theoretically lossless** — no two distinct integer-coefficient polynomials share the same value at x = e.

## How it works

1. Split the input vector into fixed-size groups (`group_size`)
2. Treat each group as polynomial coefficients
3. Evaluate at x = e → compress each group to a single number
4. Feed the reduced vector into a smaller neural network

**Constraint:** The number of input features must be exactly divisible by `group_size`.

## Results

Tested on 4 real-world datasets from scikit-learn and OpenML. No synthetic data.

| Dataset  | Normal inputs | e-Poly inputs | Time (Normal) | Time (e-Poly) | Speedup | Accuracy loss |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|
| MNIST    | 784 | 98  | 487s | 63s | **7.7x**  | −2.52% |
| USPS     | 256 | 64  | 18s  | 7s  | **2.6x**  | −1.13% |
| HAR      | 561 | 187 | 35s  | 6s  | **6.0x**  | −1.94% |
| COIL-20  | 1024| 128 | 91s  | 5s  | **16.8x** | −3.82% |

> **Note:** Results were obtained by tuning `group_size` and epoch counts. Different parameters will yield different results.

## Files

| File | Dataset | group_size | Compression |
|------|---------|-----------|-------------|
| `mainMNIST.py` | MNIST (handwritten digits) | 8 | 784 → 98 |
| `mainUSPS.py`  | USPS (postal digits) | 4 | 256 → 64 |
| `mainHAR.py`   | Human Activity Recognition | 3 | 561 → 187 |
| `mainCOIL.py`  | COIL-20 (object recognition) | 8 | 1024 → 128 |

## Requirements

```bash
pip install numpy scikit-learn
```

## Usage

```bash
python mainMNIST.py
python mainUSPS.py
python mainHAR.py
python mainCOIL.py
```

Each script trains both a standard NN and an e-Poly NN, then prints a comparison table.

## Author

**Emrecan Bayhan**  
IT Specialist & Business Analyst | PhD Student in Mechanical Engineering  
[GitHub](https://github.com/EMRECAN-1907) | [ORCID](https://orcid.org/0009-0000-2262-1498)

---

*This method may or may not exist in the literature. It came to mind, I tried it, it worked.*
