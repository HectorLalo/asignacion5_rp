"""Experimento Sonar real, Python 3.10+, ejecutar: python src/pipeline.py.

Todos los ajustes estadísticos, la selección de k y el ruido se limitan al
entrenamiento de cada fold. El test conserva su prevalencia original (UCI).
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, fbeta_score, precision_recall_curve, auc, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, StratifiedKFold as InnerCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

from cs_knn import CostKNN
from reconstruction_knn import MatrixKNN

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def load_data(path):
    data = np.genfromtxt(path, delimiter=",", dtype=str)
    if data.shape != (208, 61) or set(data[:, -1]) != {"R", "M"}:
        raise ValueError("Se requiere Sonar UCI original: 208 filas, 60 columnas, R/M")
    X = data[:, :60].astype(float)
    y = (data[:, -1] == "M").astype(int)
    assert np.isfinite(X).all() and np.all((X >= 0) & (X <= 1))
    return X, y


def undersample(X, y, ratio, rng):
    maj, minority = np.where(y == 0)[0], np.where(y == 1)[0]
    target = min(len(minority), max(1, round(len(maj) * ratio / (1 - ratio))))
    keep = np.r_[maj, rng.choice(minority, target, replace=False)]
    return X[keep], y[keep], keep


def corrupt(X, rng):
    X = X.copy()
    n = max(1, round(len(X) * .05))
    indices = rng.choice(len(X), n, replace=False)
    X[indices] += rng.normal(size=(n, X.shape[1])) * (2.5 * X.std(axis=0))
    return X, indices


class Centroid:
    def fit(self, X, y):
        self.centers = np.vstack([X[y == label].mean(axis=0) for label in (0, 1)])
        return self
    def scores(self, X):
        distances = cdist(X, self.centers)
        # Monotone margin score (not a calibrated probability).
        return distances[:, 0] - distances[:, 1]
    def predict(self, X):
        return (self.scores(X) >= 0).astype(int)


class Constant:
    def __init__(self, label): self.label=label
    def fit(self,X,y): return self
    def scores(self,X): return np.full(len(X),float(self.label))


def metrics(y, pred, scores, fn_cost):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else 0
    specificity = tn / (tn + fp) if tn + fp else 0
    precision, recall, _ = precision_recall_curve(y, scores)
    return dict(accuracy=accuracy_score(y, pred), balanced_accuracy=balanced_accuracy_score(y, pred),
                gmean=(sensitivity * specificity) ** .5, recall_mina=sensitivity,
                specificity=specificity, f1=f1_score(y, pred, zero_division=0),
                f2=fbeta_score(y, pred, beta=2, zero_division=0),
                average_precision=average_precision_score(y, scores),
                pr_auc=auc(recall, precision), roc_auc=roc_auc_score(y, scores),
                tmc=int(fn_cost * fn + fp), fn=int(fn), fp=int(fp), tp=int(tp), tn=int(tn))


def choose_k(X, y):
    # Inner CV uses only the already undersampled/noisy outer training set.
    cv = InnerCV(n_splits=3, shuffle=True, random_state=SEED)
    best = (-np.inf, 5)
    for k in (1, 3, 5, 7):
        scores = []
        for tr, val in cv.split(X, y):
            scaler = StandardScaler().fit(X[tr])
            xtr, xval = scaler.transform(X[tr]), scaler.transform(X[val])
            pca = PCA(n_components=.85, svd_solver="full").fit(xtr)
            model = KNeighborsClassifier(n_neighbors=k).fit(pca.transform(xtr), y[tr])
            scores.append(balanced_accuracy_score(y[val], model.predict(pca.transform(xval))))
        candidate = (np.mean(scores), -k)
        if candidate > best:
            best = candidate
    return -best[1]


def run(args):
    ROOT.joinpath("results").mkdir(exist_ok=True)
    data_path = ROOT / "data/sonar.all-data"
    X, y = load_data(data_path)
    sha = hashlib.sha256(data_path.read_bytes()).hexdigest()
    folds, predictions, pca_effect = [], [], []
    outer = StratifiedKFold(10, shuffle=True, random_state=SEED)
    models = ("Mínima distancia", "Naive Bayes", "KNN k=5", "CV-KNN",
              "Direct-CS-KNN", "Distances-CS-KNN", "CM-KNN", "One-Step adaptado",
              "Siempre roca", "Siempre mina")
    for fold, (itr, ite) in enumerate(outer.split(X, y), 1):
        rng = np.random.default_rng(SEED + fold)
        # The provided dataset is originally 111 M versus 97 R; sample M only in train.
        Xb, yb, keep = undersample(X[itr], y[itr], args.minority_ratio, rng)
        Xb, injected = corrupt(Xb, rng)
        scaler = StandardScaler().fit(Xb)
        Xtrain, Xtest = scaler.transform(Xb), scaler.transform(X[ite])
        projection = PCA(n_components=.85, svd_solver="full").fit(Xtrain)
        Ztr, Zte = projection.transform(Xtrain), projection.transform(Xtest)
        k = choose_k(Xb, yb)
        defs = [Centroid(), GaussianNB(), KNeighborsClassifier(n_neighbors=5),
                KNeighborsClassifier(n_neighbors=k),
                CostKNN("direct", fn_cost=args.fn_cost),
                CostKNN("distance", fn_cost=args.fn_cost),
                MatrixKNN("cm"), MatrixKNN("one"), Constant(0), Constant(1)]
        for name, model in zip(models, defs):
            t0 = time.perf_counter()
            model.fit(Ztr, yb)
            train_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            if name in ("Mínima distancia", "Direct-CS-KNN", "Distances-CS-KNN",
                        "CM-KNN", "One-Step adaptado", "Siempre roca", "Siempre mina"):
                scores = model.scores(Zte)
            elif name == "Naive Bayes" or name.startswith("KNN") or name == "CV-KNN":
                scores = model.predict_proba(Zte)[:, 1]
            if name == "Mínima distancia":
                pred = (scores >= 0).astype(int)
            elif name == "Direct-CS-KNN":
                pred = (scores >= 1 / (1 + args.fn_cost)).astype(int)
            else:
                pred = (scores >= .5).astype(int)
            infer_s = time.perf_counter() - t0
            result = metrics(y[ite], pred, scores, args.fn_cost)
            folds.append(dict(fold=fold, model=name, pca_components=Ztr.shape[1],
                              train_n=len(yb), train_mines=int(yb.sum()),
                              train_rocks=int(len(yb)-yb.sum()), injected_outliers=len(injected),
                              selected_k=k, train_seconds=train_s,
                              inference_ms_per_sample=1000*infer_s/len(ite), **result))
            predictions.extend(dict(fold=fold, original_index=int(i), label=int(true),
                                    model=name, prediction=int(p), score=float(sc))
                               for i, true, p, sc in zip(ite, y[ite], pred, scores))
        # Requested PCA ablation for Gaussian NB and KNN on the same outer split.
        for name, factory in (("Naive Bayes", GaussianNB),
                              ("KNN k=5", lambda: KNeighborsClassifier(n_neighbors=5))):
            for flag, train, test in ((0, Xtrain, Xtest), (1, Ztr, Zte)):
                obj = factory().fit(train, yb)
                p = obj.predict(test)
                pca_effect.append(dict(fold=fold, model=name, pca=flag,
                                       n_features=train.shape[1], **metrics(y[ite], p,
                                       obj.predict_proba(test)[:, 1], args.fn_cost)))
        print(f"Fold {fold}/10 | entrenamiento {len(yb)} ({yb.sum()} minas) | PCA {Ztr.shape[1]}",flush=True)
    def write(name, rows):
        with (ROOT / "results" / name).open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    write("folds.csv", folds);write("predictions.csv", predictions);write("pca_ablation.csv", pca_effect)
    summary = []
    for name in models:
        group = [x for x in folds if x["model"] == name]
        row = {"model":name}
        for key in ("accuracy","balanced_accuracy","gmean","recall_mina","specificity",
                    "f1","f2","average_precision","pr_auc","roc_auc","tmc","train_seconds","inference_ms_per_sample"):
            row[key] = float(np.mean([r[key] for r in group]))
        for key in ("tn","fp","fn","tp"):
            row[key] = sum(x[key] for x in group)
        summary.append(row)
    write("summary.csv",summary)
    consensus = []
    for i in range(len(X)):
        subset = [p for p in predictions if p["original_index"] == i and
                  p["model"] not in ("Siempre roca", "Siempre mina")]
        assert len(subset) == len(models)-2
        if all(p["label"] != p["prediction"] for p in subset):
            # Pure descriptive full-data geometry: never used to fit predictors.
            centroid_true = X[y == y[i]].mean(axis=0)
            centroid_other = X[y != y[i]].mean(axis=0)
            cov = np.cov(X[y == y[i]], rowvar=False) + np.eye(X.shape[1]) * .01
            inverse = np.linalg.inv(cov)
            def md(c):
                delta = X[i] - c
                return float(np.sqrt(delta @ inverse @ delta))
            distances = np.sort(cdist(X[i:i+1], X[y == y[i]])[0])
            consensus.append(dict(original_index=i, label=int(y[i]),
                                  mahalanobis_true=md(centroid_true),
                                  mahalanobis_other=md(centroid_other),
                                  same_class_fifth_distance=float(distances[5])))
    if consensus:write("consensus_errors.csv", consensus)
    metadata = dict(source="UCI Connectionist Bench Sonar (original)", sha256=sha,
                    n=208, rocks=int((y==0).sum()), mines=int(y.sum()),
                    outer_cv="StratifiedKFold(10,shuffle=True,random_state=42)",
                    minority_ratio_train=args.minority_ratio,
                    note="Las pruebas conservan la prevalencia original (~53% minas); sin prueba 95:5.",
                    noise="5% del entrenamiento: gaussiano 2.5 desviaciones, sin invertir etiquetas",
                    costs={"false_negative":args.fn_cost,"false_positive":1},
                    cm_and_one_step="Inferencia transductiva por lote; One-Step es adaptación con Θ fija.",
                    consensus_count=len(consensus), python=sys.version)
    (ROOT/"results/metadata.json").write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding="utf8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--minority-ratio",type=float,default=.10,choices=[.10,.05])
    parser.add_argument("--fn-cost",type=float,default=10.)
    run(parser.parse_args())
