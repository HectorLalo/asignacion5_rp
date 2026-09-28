"""Reconstrucción matricial transductiva; ninguna etiqueta de prueba se usa.

CM: objetivo publicado (Zhang et al. 2017), IRLS estabilizado. One-Step:
adaptación explícita de su objetivo con Θ proporcional a I, fija: minimización proximal de
||Y-X'W||² + α||W||²₂,₁ + β||W||²₁. La versión completa del artículo optimiza
Θ y usa grupos: esta implementación NO se presenta como reproducción exacta.
"""
import numpy as np
from scipy.linalg import solve
from scipy.spatial.distance import cdist


class MatrixKNN:
    def __init__(self, mode="cm", alpha=.15, beta=.02, laplacian=.01,
                 iterations=12, tolerance=1e-5):
        self.mode, self.alpha, self.beta = mode, alpha, beta
        self.laplacian, self.iterations, self.tolerance = laplacian, iterations, tolerance

    def fit(self, X, y):
        self.X, self.y = np.asarray(X, float).copy(), np.asarray(y, int).copy()
        n, d = self.X.shape
        self.gram = self.X @ self.X.T
        self.ridge = np.eye(n) * 1e-5
        if self.mode == "cm":
            # Grafo no supervisado de características, ajustado SOLO en entrenamiento.
            feats = self.X.T
            dist = cdist(feats, feats)
            s = np.zeros((d, d))
            scale = np.median(dist[dist > 0]) if np.any(dist > 0) else 1.
            for i in range(d):
                idx = np.argsort(dist[i])[1:min(6, d)]
                s[i, idx] = np.exp(-dist[i, idx] ** 2 / (scale ** 2 + 1e-12))
            s = np.maximum(s, s.T)
            lap = np.diag(s.sum(axis=0)) - s
            self.base = self.gram + self.laplacian * self.X @ lap @ self.X.T
        return self

    def weights(self, Y):
        Y = np.asarray(Y, float)
        n, m = len(self.X), len(Y)
        B = self.X @ Y.T
        W = solve(self.gram + np.eye(n) * .5, B, assume_a="pos")
        if self.mode == "cm":
            for _ in range(self.iterations):
                prev = W.copy()
                rows = 1 / (2 * np.maximum(np.linalg.norm(W, axis=1), 1e-4))
                for j in range(m):
                    elem = 1 / (2 * np.maximum(np.abs(W[:, j]), 1e-4))
                    H = self.base + np.diag(self.alpha * elem + self.beta * rows) + self.ridge
                    W[:, j] = solve(H, B[:, j], assume_a="pos")
                if np.linalg.norm(W - prev) / (np.linalg.norm(prev) + 1e-9) < self.tolerance:
                    break
        else:
            # Proximal gradient for the stated squared mixed norms, W>=0 for votes.
            L = 2 * np.linalg.norm(self.gram, ord=2) + 1e-8
            step = 1 / L
            W = np.maximum(W, 0)
            for _ in range(self.iterations * 8):
                prev = W.copy()
                rn = np.linalg.norm(W, axis=1)
                Z = W - step * (2 * (self.gram @ W - B) +
                                2 * self.alpha * rn.sum() * W / np.maximum(rn[:, None], 1e-8) +
                                2 * self.beta * W.sum() * np.sign(W))
                W = np.maximum(Z, 0)
                if np.linalg.norm(W - prev) / (np.linalg.norm(prev) + 1e-9) < self.tolerance:
                    break
        return W

    def scores(self, Y):
        W = np.maximum(self.weights(Y), 0)
        den = W.sum(axis=0)
        result = np.divide(self.y @ W, den, out=np.zeros(W.shape[1]), where=den > 1e-10)
        fallback = den <= 1e-10
        if fallback.any():
            idx = cdist(np.asarray(Y)[fallback], self.X).argmin(axis=1)
            result[fallback] = self.y[idx]
        return result

    def predict(self, Y):
        return (self.scores(Y) >= .5).astype(int)
