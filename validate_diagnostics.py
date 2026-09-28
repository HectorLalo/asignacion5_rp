"""Verifica el AUC mediante comparación de pares, independientemente de sklearn."""
import csv
from pathlib import Path
import numpy as np
from scipy.integrate import trapezoid

ROOT=Path(__file__).resolve().parent

def read(name):
    with (ROOT/'results'/name).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def audit():
    pred=read('predictions.csv');aucs=read('roc_auc_folds.csv');points=read('roc_points.csv');stats=read('descriptive_statistics.csv');cms=read('confusion_matrices.csv')
    assert len(aucs)==100 and len(cms)==40
    models=sorted({r['model'] for r in pred})
    for model in models:
        allrows=[r for r in pred if r['model']==model]
        assert len(allrows)==208
        for real in (0,1):
            rr=[r for r in cms if r['model']==model and int(r['real'])==real]
            assert sum(int(r['count']) for r in rr)==(97 if real==0 else 111)
            assert np.isclose(sum(float(r['row_percent']) for r in rr),100)
            for r in rr:
                count=sum(int(p['label'])==real and int(p['prediction'])==int(r['predicted']) for p in allrows)
                assert count==int(r['count'])
        values=[]
        for fold in range(1,11):
            rows=[r for r in allrows if int(r['fold'])==fold]
            pos=np.array([float(r['score']) for r in rows if int(r['label'])==1]);neg=np.array([float(r['score']) for r in rows if int(r['label'])==0])
            # Probabilidad de que un positivo supere un negativo; empates valen 1/2.
            rank_auc=np.mean((pos[:,None]>neg).astype(float)+.5*(pos[:,None]==neg))
            a=next(r for r in aucs if r['model']==model and int(r['fold'])==fold)
            assert np.isclose(rank_auc,float(a['roc_auc']),atol=1e-12)
            curve=[r for r in points if r['model']==model and int(r['fold'])==fold]
            x=np.array([float(r['fpr']) for r in curve]);y=np.array([float(r['tpr']) for r in curve])
            assert (np.diff(x)>=0).all() and (np.diff(y)>=0).all()
            assert np.isclose(trapezoid(y,x),rank_auc,atol=1e-12)
            values.append(rank_auc)
        s=next(r for r in stats if r['model']==model and r['metric']=='roc_auc')
        assert np.isclose(float(s['mean']),np.mean(values)) and np.isclose(float(s['std']),np.std(values,ddof=1))
        if model.startswith('Siempre'):assert np.allclose(values,.5)
    for name in ['confusion_referencias','confusion_variantes','confusion_controles','roc_referencias','roc_variantes','auc_comparacion','ba_variabilidad','mapa_metricas','diferencias_pareadas','pca_pareado']:
        for ext in ['png','pdf']:assert (ROOT/'figures'/f'{name}.{ext}').is_file()
    print('VALIDADO: 100 AUC por pares y por area ROC, 40 celdas de confusion, porcentajes, medias y DE.')

if __name__=='__main__':audit()
