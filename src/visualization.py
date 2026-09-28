"""Figuras reproducibles desde results/; ejecutar tras pipeline.py."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from cs_knn import CostKNN
from reconstruction_knn import MatrixKNN
from pipeline import ROOT, load_data, undersample, corrupt, Centroid


def read(name):
    with (ROOT / "results" / name).open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def create():
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10,
                         "axes.spines.top":False,"axes.spines.right":False})
    out=ROOT / "figures";out.mkdir(exist_ok=True)
    rows=read("summary.csv")
    names=[r["model"].replace("Mínima distancia","Mín. distancia").replace("Distances-","Dist.-") for r in rows]
    fields=[("balanced_accuracy","Exactitud balanceada"),("gmean","G-Mean"),
            ("f1","F1"),("f2","F2"),("pr_auc","PR-AUC")]
    fig,axes=plt.subplots(1,2,figsize=(14,5.5),layout="constrained")
    x=np.arange(len(rows)); width=.16
    for j,(key,label) in enumerate(fields):
        axes[0].bar(x+(j-2)*width,[float(r[key]) for r in rows],width,label=label)
    axes[0].set(xticks=x,xticklabels=names,ylim=(0,1.07),ylabel="Valor (0–1)",title="Todas las métricas; 10 folds")
    axes[0].tick_params(axis="x",rotation=63,labelsize=8)
    axes[0].legend(fontsize=8,ncol=2)
    axes[1].bar(x-.18,[float(r["fn"]) for r in rows],.35,label="FN minas",color="#B34045")
    axes[1].bar(x+.18,[float(r["fp"]) for r in rows],.35,label="FP rocas",color="#1B396A")
    axes[1].set(xticks=x,xticklabels=names,ylabel="Total de errores (208 predicciones)",title="Errores por clase")
    axes[1].tick_params(axis="x",rotation=63,labelsize=8);axes[1].legend(fontsize=8)
    fig.savefig(out/"contraste_metricas.png",dpi=175);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.3),layout="constrained")
    axes[0].barh(names,[float(r["tmc"]) for r in rows],color="#1B396A")
    axes[0].set(xlabel="TMC medio por fold (10·FN + FP)",title="Costo (menor es mejor)")
    axes[1].barh(names,[float(r["inference_ms_per_sample"]) for r in rows],color="#B34045")
    axes[1].set(xlabel="Milisegundos por muestra",title="Inferencia; incluye reconstrucción matricial")
    fig.savefig(out/"costo_latencia.png",dpi=175);plt.close(fig)
    pca=read("pca_ablation.csv")
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained")
    for ax,key,title in zip(axes,["balanced_accuracy","pr_auc"],["Exactitud balanceada","PR-AUC"]):
        for model,style in [("Naive Bayes","o-"),("KNN k=5","s-")]:
            yy=[np.mean([float(r[key]) for r in pca if r["model"]==model and int(r["pca"])==flag]) for flag in (0,1)]
            ax.plot([0,1],yy,style,label=model)
        ax.set(xlabel="PCA",ylabel=title,xticks=[0,1],xticklabels=["60 originales","≥85% varianza"])
        ax.legend(fontsize=9)
    fig.savefig(out/"ablacion_pca.png",dpi=175);plt.close(fig)

    # Visualización descriptiva de 2 componentes; NO son las métricas anteriores.
    X,y=load_data(ROOT/"data/sonar.all-data")
    rng=np.random.default_rng(42)
    xb,yb,_=undersample(X,y,.10,rng);xb,_=corrupt(xb,rng)
    scaler=StandardScaler().fit(xb); p=PCA(n_components=2).fit(scaler.transform(xb))
    tr=p.transform(scaler.transform(xb));allx=p.transform(scaler.transform(X))
    lo=allx.min(axis=0)-1;hi=allx.max(axis=0)+1
    gx,gy=np.meshgrid(np.linspace(lo[0],hi[0],22),np.linspace(lo[1],hi[1],22))
    grid=np.c_[gx.ravel(),gy.ravel()]
    defs=[Centroid(),GaussianNB(),KNeighborsClassifier(n_neighbors=5),
          KNeighborsClassifier(n_neighbors=3),CostKNN("direct"),CostKNN("distance"),
          MatrixKNN("cm",iterations=7),MatrixKNN("one",iterations=7)]
    fig,axes=plt.subplots(2,4,figsize=(15,7),layout="constrained")
    for ax,r,model in zip(axes.flat,rows,defs):
        model.fit(tr,yb)
        z=model.predict(grid).reshape(gx.shape)
        ax.contourf(gx,gy,z,levels=[-.5,.5,1.5],colors=["#C6D5E6","#F3C4C1"],alpha=.85)
        ax.scatter(allx[y==0,0],allx[y==0,1],s=12,c="#1B396A",marker="o",label="R")
        ax.scatter(allx[y==1,0],allx[y==1,1],s=16,c="#A83236",marker="^",label="M")
        ax.set(title=r["model"],xlabel="CP 1",ylabel="CP 2")
    fig.suptitle("Fronteras ilustrativas en 2 componentes, distintos del benchmark ≥85%",fontsize=13)
    fig.savefig(out/"fronteras_2d.png",dpi=175);plt.close(fig)

if __name__=="__main__":create()
