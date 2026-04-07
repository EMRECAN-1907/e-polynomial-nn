"""
mainHAR.py — Human Activity Recognition
Normal NN vs e-Poly NN
Giriş : 561 özellik (sensör verisi)
Çıkış : 6 sınıf (yürüme, merdiven, oturma, ayakta, yatma, iniş)
e-Poly: group_size=11 → 561/11 = 51 giriş
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import time

np.random.seed(42)
e = np.e

# ── e-POLİNOM ─────────────────────────────────────────────────────────────────
def e_compress(X, group_size=11):
    n, d = X.shape
    n_groups = d // group_size
    powers = np.array([e**i for i in range(group_size - 1, -1, -1)])
    return X.reshape(n, n_groups, group_size) @ powers

# ── VERİ ─────────────────────────────────────────────────────────────────────
print("HAR verisi yükleniyor...")
_data = fetch_openml('har', version=1, as_frame=False, parser='liac-arff')
X_ham = _data.data.astype(float)
_le   = LabelEncoder()
y     = _le.fit_transform(_data.target)

print(f"Veri boyutu: {X_ham.shape}  |  Sınıf sayısı: {len(np.unique(y))}")

X_comp = e_compress(X_ham, group_size=3)  # 561 → 51
print(f"e-Poly boyutu: {X_comp.shape}")

X_tr,   X_te,   y_tr, y_te = train_test_split(X_ham,  y, test_size=0.2, random_state=42, stratify=y)
X_c_tr, X_c_te, _,    _    = train_test_split(X_comp, y, test_size=0.2, random_state=42, stratify=y)

sx_n = MinMaxScaler(); X_n_tr = sx_n.fit_transform(X_tr);   X_n_te = sx_n.transform(X_te)
sx_c = MinMaxScaler(); X_c_tr = sx_c.fit_transform(X_c_tr); X_c_te = sx_c.transform(X_c_te)

N_SINIF = 6

# ── NN FONKSİYONLARI ──────────────────────────────────────────────────────────
def relu(z):   return np.maximum(0, z)
def relu_d(z): return (z > 0).astype(float)
def softmax(z):
    ex = np.exp(z - z.max(axis=0, keepdims=True))
    return ex / ex.sum(axis=0, keepdims=True)

def init(dims):
    p = {}
    for l in range(1, len(dims)):
        p[f"W{l}"] = np.random.randn(dims[l], dims[l-1]) * np.sqrt(2/dims[l-1])
        p[f"b{l}"] = np.zeros((dims[l], 1))
    return p

def forward(X, p, L):
    c = {"A0": X.T}
    for l in range(1, L):
        Z = p[f"W{l}"] @ c[f"A{l-1}"] + p[f"b{l}"]
        A = relu(Z) if l < L-1 else softmax(Z)
        c.update({f"Z{l}": Z, f"A{l}": A})
    return c[f"A{L-1}"], c

def backward(p, c, y, L):
    m = len(y); g = {}
    Y = np.zeros((N_SINIF, m)); Y[y, np.arange(m)] = 1
    dA = c[f"A{L-1}"] - Y
    for l in range(L-1, 0, -1):
        dZ = dA if l == L-1 else dA * relu_d(c[f"Z{l}"])
        g[f"dW{l}"] = dZ @ c[f"A{l-1}"].T / m
        g[f"db{l}"] = np.sum(dZ, axis=1, keepdims=True) / m
        dA = p[f"W{l}"].T @ dZ
    return g

def update(p, g, lr, L, l2=0.001):
    for l in range(1, L):
        p[f"W{l}"] -= lr * (g[f"dW{l}"] + l2 * p[f"W{l}"])
        p[f"b{l}"] -= lr * g[f"db{l}"]
    return p

def acc(X, y, p, L):
    A, _ = forward(X, p, L)
    return np.mean(A.argmax(axis=0) == y) * 100

def n_params(p, L):
    return sum(p[f"W{l}"].size + p[f"b{l}"].size for l in range(1, L))

def egit(X_tr, y_tr, X_te, y_te, dims, epochs, lr=0.01, bs=128, label=""):
    L = len(dims); p = init(dims); m = X_tr.shape[0]
    print(f"\n{label} eğitiliyor... ({dims[0]} giriş, {epochs} epoch)\n")
    t0 = time.time()
    for ep in range(1, epochs + 1):
        idx = np.random.permutation(m)
        for i in range(0, m, bs):
            Xb, yb = X_tr[idx[i:i+bs]], y_tr[idx[i:i+bs]]
            A, c = forward(Xb, p, L)
            g = backward(p, c, yb, L)
            p = update(p, g, lr, L)
        if ep % 50 == 0:
            print(f"  Epoch {ep:3d}  |  Train: %{acc(X_tr, y_tr, p, L):.2f}  |  Test: %{acc(X_te, y_te, p, L):.2f}")
    return p, time.time()-t0, n_params(p, L), L

# ── ÇALIŞTIR ──────────────────────────────────────────────────────────────────
p_n, t_n, par_n, L_n = egit(X_n_tr, y_tr, X_n_te, y_te, [561, 128, 64, N_SINIF], 200, label="Normal NN")
p_c, t_c, par_c, L_c = egit(X_c_tr, y_tr, X_c_te, y_te, [187, 128, 64, N_SINIF], 300, label="e-Poly NN")

# ── SONUÇLAR ──────────────────────────────────────────────────────────────────
print("\n" + "═"*56)
print(f"  {'Metrik':<22} {'Normal NN':>14} {'e-Poly NN':>14}")
print("─"*56)
print(f"  {'Giriş nöronu':<22} {561:>14} {187:>14}")
print(f"  {'Parametre sayısı':<22} {par_n:>14} {par_c:>14}")
print(f"  {'Epoch':<22} {200:>14} {300:>14}")
print(f"  {'Süre (sn)':<22} {t_n:>14.2f} {t_c:>14.2f}")
print(f"  {'Test doğruluğu (%)':<22} {acc(X_n_te, y_te, p_n, L_n):>14.2f} {acc(X_c_te, y_te, p_c, L_c):>14.2f}")
print("═"*56)