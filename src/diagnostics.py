"""Matrices, ROC/AUC y estadística descriptiva a partir de predicciones externas.

No reentrena ni selecciona umbrales. M=1 es la clase positiva.
ROC/AUC se calcula dentro de cada pliegue. Se promedian curvas interpoladas,
no los puntajes de modelos distintos. La DE entre pliegues no es un IC.
Ejecutar: python src/diagnostics.py. Exporta PNG/PDF y CSV auditables.
"""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
BLUE, RED = '#1B396A', '#B34045'
COLORS = ['#1B396A','#BE6435','#258477','#905D9C']

def read(name):
    with (ROOT/'results'/name).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def write(name, rows):
    with (ROOT/'results'/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)

def short(model):
    return model.replace('Mínima distancia','Mín. distancia').replace('Distances-CS-KNN','Dist.-CS').replace('Direct-CS-KNN','Direct-CS').replace('One-Step adaptado','One-Step*')

def save(fig,name):
    for ext in ('png','pdf'):
        fig.savefig(ROOT/'figures'/f'{name}.{ext}',dpi=190,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def create():
    (ROOT/'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.titlesize':15,
        'axes.labelsize':12,'axes.spines.top':False,'axes.spines.right':False,'legend.fontsize':10})
    preds=read('predictions.csv');folds=read('folds.csv');summary=read('summary.csv')
    models=[r['model'] for r in summary];counts=[];stats=[];curves=[];auc_rows=[]
    mean_grid=np.linspace(0,1,501);cache={};matrices={}
    for model in models:
        rows=[r for r in preds if r['model']==model]
        assert len(rows)==208 and len({r['original_index'] for r in rows})==208
        yy=np.array([int(r['label']) for r in rows]);pp=np.array([int(r['prediction']) for r in rows])
        cm=confusion_matrix(yy,pp,labels=[0,1]);matrices[model]=cm
        for i in range(2):
            for j in range(2):
                counts.append(dict(model=model,real=i,predicted=j,count=int(cm[i,j]),row_percent=100*cm[i,j]/cm[i].sum()))
        folddata=[]
        for fold in range(1,11):
            part=[r for r in rows if int(r['fold'])==fold]
            y=np.array([int(r['label']) for r in part]);p=np.array([int(r['prediction']) for r in part]);score=np.array([float(r['score']) for r in part])
            assert set(y)=={0,1} and np.isfinite(score).all()
            fpr,tpr,thr=roc_curve(y,score,drop_intermediate=False)
            auc=float(roc_auc_score(y,score));pr,rec,_=precision_recall_curve(y,score)
            for x,t,h in zip(fpr,tpr,thr):curves.append(dict(model=model,fold=fold,fpr=float(x),tpr=float(t),threshold=float(h)))
            tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
            auc_rows.append(dict(model=model,fold=fold,roc_auc=auc,average_precision=average_precision_score(y,score),fpr_at_rule=fp/(fp+tn),tpr_at_rule=tp/(tp+fn)))
            interp=np.interp(mean_grid,fpr,tpr);interp[0]=0;interp[-1]=1
            folddata.append(dict(fpr=fpr,tpr=tpr,interp=interp,auc=auc,precision=pr,recall=rec))
        cache[model]=folddata
        fr=[r for r in folds if r['model']==model]
        for key in ['roc_auc','balanced_accuracy','gmean','f1','f2','average_precision','pr_auc','tmc','inference_ms_per_sample']:
            vals=np.array([d['auc'] for d in folddata]) if key=='roc_auc' else np.array([float(r[key]) for r in fr])
            q1,median,q3=np.quantile(vals,[.25,.5,.75])
            stats.append(dict(model=model,metric=key,n=10,mean=vals.mean(),std=vals.std(ddof=1),median=median,q1=q1,q3=q3,min=vals.min(),max=vals.max()))
    write('confusion_matrices.csv',counts);write('roc_points.csv',curves);write('roc_auc_folds.csv',auc_rows);write('descriptive_statistics.csv',stats)
    means=[]
    for model in models:
        vals=np.array([d['interp'] for d in cache[model]])
        for x,m,s in zip(mean_grid,vals.mean(axis=0),vals.std(axis=0,ddof=1)):
            means.append(dict(model=model,fpr=x,tpr_mean=m,tpr_std=s))
    write('roc_mean_curves.csv',means)
    sm={(r['model'],r['metric']):r for r in stats}
    # Mismo color significa el mismo porcentaje por fila, no la misma cantidad.
    def matrix_figure(group,name):
        cols=2;nr=(len(group)+1)//2
        fig,axes=plt.subplots(nr,cols,figsize=(11.2,4.0*nr),squeeze=False,layout='constrained')
        for ax,model in zip(axes.flat,group):
            cm=matrices[model];pct=100*cm/cm.sum(axis=1,keepdims=True)
            ax.imshow(pct,vmin=0,vmax=100,cmap='Blues')
            labels=[['TN','FP'],['FN','TP']]
            for i in range(2):
                for j in range(2):
                    ax.text(j,i,f'{labels[i][j]}\n{cm[i,j]}  ({pct[i,j]:.1f}%)',ha='center',va='center',fontsize=13,color='white' if pct[i,j]>65 else '#172B45')
            ax.set(xticks=[0,1],xticklabels=['Roca','Mina'],yticks=[0,1],yticklabels=['Roca (97)','Mina (111)'],xlabel='Clase predicha',ylabel='Clase real',title=short(model))
        for ax in list(axes.flat)[len(group):]:ax.set_visible(False)
        fig.suptitle('208 predicciones externas por método; porcentaje dentro de cada clase real',fontsize=14)
        save(fig,name)
    matrix_figure(models[:4],'confusion_referencias');matrix_figure(models[4:8],'confusion_variantes');matrix_figure(models[8:],'confusion_controles')
    matrix_figure(['KNN k=5','Direct-CS-KNN'],'confusion_knn_direct')
    matrix_figure(['Distances-CS-KNN','Siempre mina'],'confusion_distancias_control')
    for gi,group in enumerate([models[:4],models[4:8]]):
        fig,ax=plt.subplots(figsize=(10.5,5.5),layout='constrained')
        for i,model in enumerate(group):
            roc=np.mean([d['interp'] for d in cache[model]],axis=0);a=sm[model,'roc_auc']
            ax.plot(mean_grid,roc,color=COLORS[i],lw=2.4,linestyle=['-','--','-.',':'][i],label=f'{short(model)}: {a["mean"]:.3f} ± {a["std"]:.3f}')
        ax.plot([0,1],[0,1],color='#888888',lw=1.3,ls='--',label='Control constante: AUC = 0.500')
        ax.set(xlim=(0,1),ylim=(0,1.02),xlabel='Falsas alarmas / rocas (FPR)',ylabel='Minas detectadas / minas (TPR)',title='ROC media; leyenda: AUC medio ± DE de 10 pliegues')
        ax.grid(alpha=.18);ax.legend(loc='lower right',framealpha=.95)
        save(fig,'roc_referencias' if gi==0 else 'roc_variantes')
        # Curvas PR por pliegue: evita equiparar AP a integral de una curva promedio.
        fig,axs=plt.subplots(2,2,figsize=(11.2,7.2),layout='constrained')
        for ax,model,col in zip(axs.flat,group,COLORS):
            for d in cache[model]:ax.step(d['recall'][::-1],d['precision'][::-1],where='pre',alpha=.35,color=col)
            a=sm[model,'average_precision'];ax.axhline(111/208,ls=':',color='#777777')
            ax.set(xlim=(0,1),ylim=(0,1.02),xlabel='Sensibilidad',ylabel='Precisión',title=f'{short(model)} · AP={a["mean"]:.3f}');ax.grid(alpha=.15)
        fig.suptitle('Cada línea es un pliegue; línea gris: prevalencia global 111/208 (orientativa)',fontsize=13)
        save(fig,'pr_referencias' if gi==0 else 'pr_variantes')
    for key,name,title in [('roc_auc','auc_comparacion','AUC-ROC por método'),('balanced_accuracy','ba_variabilidad','Exactitud balanceada por método')]:
        fig,ax=plt.subplots(figsize=(11.4,6.2),layout='constrained')
        for i,model in enumerate(models):
            vals=np.array([d['auc'] for d in cache[model]]) if key=='roc_auc' else np.array([float(r[key]) for r in folds if r['model']==model])
            ax.scatter(vals,i+np.linspace(-.15,.15,10),s=28,color=BLUE,alpha=.5)
            ax.errorbar(vals.mean(),i,xerr=vals.std(ddof=1),fmt='D',color=RED,capsize=4,ms=6)
        ax.set(yticks=range(10),yticklabels=[short(m) for m in models],xlim=(-.02,1.02),xlabel='Puntos: pliegues. Diamante y barra: media ± DE (no IC)',title=title)
        ax.invert_yaxis();ax.axvline(.5,ls=':',color='#777777');ax.grid(axis='x',alpha=.2);save(fig,name)
    # Contrastes pareados por las mismas particiones; no inferencia estadística.
    contrasts=[]
    fig,axes=plt.subplots(1,2,figsize=(11.6,4.6),layout='constrained')
    base={int(r['fold']):r for r in folds if r['model']=='KNN k=5'}
    for ax,key,title in zip(axes,['balanced_accuracy','tmc'],['Cambio de BA (más alto favorece)','Cambio de costo (más bajo favorece)']):
        for i,model in enumerate(models[4:8]):
            fr=sorted([r for r in folds if r['model']==model],key=lambda x:int(x['fold']))
            diff=np.array([float(r[key])-float(base[int(r['fold'])][key]) for r in fr])
            ax.scatter(diff,i+np.linspace(-.14,.14,10),color=COLORS[i],alpha=.65,s=28)
            ax.plot(diff.mean(),i,'D',color='black',ms=7)
            contrasts.extend(dict(model=model,fold=int(r['fold']),metric=key,difference=float(v)) for r,v in zip(fr,diff))
        ax.axvline(0,color='#555555',ls='--');ax.set(yticks=range(4),yticklabels=[short(m) for m in models[4:8]],xlabel='Cada punto: diferencia en un mismo pliegue',title=title);ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
    save(fig,'diferencias_pareadas');write('paired_differences.csv',contrasts)
    metrics=['accuracy','balanced_accuracy','gmean','f1','f2','average_precision','pr_auc','roc_auc']
    values=np.array([[float(r[k]) for k in metrics] for r in summary])
    fig,ax=plt.subplots(figsize=(11.6,6.1),layout='constrained');ax.imshow(values,vmin=0,vmax=1,cmap='Blues',aspect='auto')
    for i in range(10):
        for j in range(8):ax.text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',color='white' if values[i,j]>.66 else BLUE,fontsize=13)
    ax.set(xticks=range(8),xticklabels=['Acc','BA','G','F1','F2','AP','PR-AUC','ROC-AUC'],yticks=range(10),yticklabels=[short(m) for m in models],title='Medias de 10 pliegues; cada columna responde una pregunta distinta')
    save(fig,'mapa_metricas')
    # Efecto pareado de PCA por modelo, usando observaciones de los mismos pliegues.
    ab=read('pca_ablation.csv');fig,axs=plt.subplots(1,2,figsize=(10.5,4.5),layout='constrained')
    for ax,model in zip(axs,['Naive Bayes','KNN k=5']):
        pairs=[]
        for fold in range(1,11):
            pair=[float(next(r for r in ab if r['model']==model and int(r['fold'])==fold and int(r['pca'])==p)['balanced_accuracy']) for p in (0,1)]
            pairs.append(pair);ax.plot([0,1],pair,'o-',color=BLUE,alpha=.28)
        ax.plot([0,1],np.mean(pairs,axis=0),'D-',color=RED,lw=3,ms=8,label='Media')
        ax.set(xticks=[0,1],xticklabels=['60 atributos','PCA ≥85%'],ylim=(.45,.82),ylabel='Exactitud balanceada',title=model);ax.grid(alpha=.15);ax.legend()
    save(fig,'pca_pareado')
    metadata={'positive_class':'M=1','n_predictions_per_model':208,'n_folds':10,
        'auc':'media aritmetica del AUC exacto en cada pliegue; no area de curva media ni AUC de scores agrupados',
        'roc_mean':'interpolacion de TPR en 501 valores FPR [0,1]; extremos (0,0) y (1,1)',
        'spread':'DE muestral (ddof=1). No IC ni prueba de significancia; folds comparten entrenamiento.',
        'confusion':'conteos sumados; porcentaje por fila (clase real)',
        'no_threshold_selection':True}
    (ROOT/'results/diagnostics_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print('DIAGNOSTICOS: matrices, ROC, PR, AUC, dispersion y diferencias pareadas exportadas a PNG/PDF/CSV.')

if __name__=='__main__':create()
