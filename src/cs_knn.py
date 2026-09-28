"""Vecinos sensibles al costo. Clases: 0=roca, 1=mina.

Implementaciones explícitas de las reglas del anexo B; el escalamiento de
distancias es una adaptación parametrizada, no una réplica numérica del artículo.
"""
import numpy as np
from scipy.spatial.distance import cdist


class CostKNN:
    def __init__(self, mode="direct", k=5, fn_cost=10., fp_cost=1., m=2., gamma=.3):
        self.mode, self.k = mode, k
        self.fn_cost, self.fp_cost, self.m, self.gamma = fn_cost, fp_cost, m, gamma

    def fit(self, X, y):
        self.X = np.asarray(X, float).copy()
        self.y = np.asarray(y, int).copy()
        assert set(np.unique(self.y)) == {0, 1} and 1 <= self.k <= len(y)
        self.prior = self.y.mean()
        return self

    def scores(self, X):
        d = cdist(np.asarray(X, float), self.X)
        if self.mode == "distance":
            ratio = (self.fn_cost / self.fp_cost) * (sum(self.y == 0) / sum(self.y == 1))
            d[:, self.y == 1] /= ratio ** self.gamma
        idx = np.argsort(d, axis=1, kind="stable")[:, :self.k]
        positives = self.y[idx].sum(axis=1)
        if self.mode == "direct":
            return (positives + self.m * self.prior) / (self.k + self.m)
        return positives / self.k

    def predict(self, X):
        threshold = self.fp_cost / (self.fn_cost + self.fp_cost) if self.mode == "direct" else .5
        return (self.scores(X) >= threshold).astype(int)
