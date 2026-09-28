"""Auditoría independiente de predicciones, folds, costo y resumen."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_recall_curve, auc, f1_score,
                             fbeta_score, confusion_matrix, roc_auc_score)

ROOT=Path(__file__).resolve().parent
def read(name):
    with (ROOT/"results"/name).open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def audit():
    raw=(ROOT/"data/sonar.all-data").read_bytes()
    meta=json.loads((ROOT/"results/metadata.json").read_text())
    assert hashlib.sha256(raw).hexdigest()==meta['sha256']
    a=np.genfromtxt(ROOT/"data/sonar.all-data",delimiter=",",dtype=str)
    y=(a[:,-1]=="M").astype(int)
    assert a.shape==(208,61) and Counter(y)=={0:97,1:111}
    predictions=read("predictions.csv");folds=read("folds.csv");summary=read("summary.csv")
    models={r["model"] for r in summary}
    assert len(models)==10 and len(predictions)==2080 and len(folds)==100
    splits=list(StratifiedKFold(10,shuffle=True,random_state=42).split(a[:,:60],y))
    grouped=defaultdict(list)
    for row in predictions:
        model,fold,index,label=row['model'],int(row['fold']),int(row['original_index']),int(row['label'])
        assert index in splits[fold-1][1] and label==y[index]
        grouped[(model,fold)].append(row)
    for model in models:
        assert sorted(int(r['original_index']) for r in predictions if r['model']==model)==list(range(208))
    for row in folds:
        key=(row['model'],int(row['fold']))
        part=grouped[key]
        assert len(part)==len(splits[key[1]-1][1])
        true=np.array([int(p['label']) for p in part]);pred=np.array([int(p['prediction']) for p in part]);score=np.array([float(p['score']) for p in part])
        tn,fp,fn,tp=confusion_matrix(true,pred,labels=[0,1]).ravel()
        pr,rc,_=precision_recall_curve(true,score)
        checks={'tn':tn,'fp':fp,'fn':fn,'tp':tp,'tmc':10*fn+fp,
                'accuracy':accuracy_score(true,pred),
                'balanced_accuracy':balanced_accuracy_score(true,pred),
                'f1':f1_score(true,pred,zero_division=0),
                'f2':fbeta_score(true,pred,beta=2,zero_division=0),
                'pr_auc':auc(rc,pr), 'roc_auc':roc_auc_score(true,score)}
        for k,v in checks.items():assert np.isclose(float(row[k]),v,atol=1e-9),(key,k)
    for row in summary:
        part=[f for f in folds if f['model']==row['model']]
        assert len(part)==10
        for key in ['accuracy','balanced_accuracy','gmean','f1','f2','pr_auc','roc_auc','average_precision','tmc']:
            assert np.isclose(float(row[key]),np.mean([float(x[key]) for x in part]))
        for key in ['tn','fp','fn','tp']:
            assert int(row[key])==sum(int(x[key]) for x in part)
        assert sum(int(row[k]) for k in ['tn','fp','fn','tp'])==208
    print("VALIDADO: Sonar auténtico, 10 folds, 208 índices/modelo, etiquetas, confusiones, métricas, costos y promedios.")

if __name__=='__main__': audit()
