"""Diagnóstico descriptivo de ejemplos que fallan en 5+ de 8 modelos.
Nunca se usan estos estadísticos para entrenar ni seleccionar hiperparámetros.
"""
import csv
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pipeline import ROOT, load_data


def run():
    with (ROOT/'results/predictions.csv').open(encoding='utf-8-sig') as f:
        predictions=list(csv.DictReader(f))
    by=defaultdict(list)
    for r in predictions:
        if not r['model'].startswith('Siempre'):
            by[int(r['original_index'])].append(r)
    X,y=load_data(ROOT/'data/sonar.all-data')
    scaled=StandardScaler().fit_transform(X)  # Solo diagnóstico posterior.
    pooled=LedoitWolf().fit(scaled).precision_
    centroids=np.vstack([scaled[y==c].mean(axis=0) for c in (0,1)])
    out=[]
    for i in range(len(X)):
        err=sum(int(r['label'])!=int(r['prediction']) for r in by[i])
        same=np.sort(cdist(scaled[i:i+1],scaled[y==y[i]])[0])
        def md(c):
            delta=scaled[i]-centroids[c]
            return np.sqrt(max(0,float(delta@pooled@delta)))
        out.append(dict(index=i,label=int(y[i]),errors=err,hard=int(err>=5),
                        mahalanobis_true=md(y[i]),mahalanobis_other=md(1-y[i]),
                        same_class_fifth_distance=float(same[5])))
    with (ROOT/'results/hard_cases.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=out[0]);writer.writeheader();writer.writerows(out)
    hard=np.array([r['hard'] for r in out],bool)
    pca=PCA(n_components=2).fit_transform(scaled)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    axes[0].scatter(pca[y==0,0],pca[y==0,1],c='#1B396A',s=17,label='Rocas')
    axes[0].scatter(pca[(y==1)&~hard,0],pca[(y==1)&~hard,1],c='#B34045',s=22,marker='^',label='Minas, <5 errores')
    axes[0].scatter(pca[hard,0],pca[hard,1],facecolors='none',edgecolors='black',s=95,linewidths=1.3,label='≥5 errores')
    axes[0].set(xlabel='CP 1',ylabel='CP 2',title='65 casos difíciles; todos minas')
    axes[0].legend(fontsize=7,loc='upper right')
    mine=[r for r in out if r['label']==1]
    for ax,key,title in ((axes[1],'mahalanobis_true','Mahalanobis al centro mina'),
                         (axes[2],'same_class_fifth_distance','Distancia al 5º vecino mina')):
        vals=[[r[key] for r in mine if r['hard']==flag] for flag in (0,1)]
        ax.boxplot(vals,tick_labels=['<5 errores','≥5 errores'],showfliers=False)
        ax.set(title=title,ylabel='Distancia estandarizada')
    fig.savefig(ROOT/'figures/hard_cases.png',dpi=190);plt.close(fig)
    print('Casos ≥5 de 8:',hard.sum(),'Rocas:',sum(hard&(y==0)),'Minas:',sum(hard&(y==1)))
    print('Medianas Mahalanobis mina: difícil',round(np.median([r['mahalanobis_true'] for r in mine if r['hard']]),3),
          'resto',round(np.median([r['mahalanobis_true'] for r in mine if not r['hard']]),3))
    print('Medianas distancia 5º: difícil',round(np.median([r['same_class_fifth_distance'] for r in mine if r['hard']]),3),
          'resto',round(np.median([r['same_class_fifth_distance'] for r in mine if not r['hard']]),3))

if __name__=='__main__':run()
