"""Reporte reproducible de hasta 10 páginas: texto en dos columnas y figuras legibles."""
import csv
from pathlib import Path
from xml.sax.saxutils import escape
from PIL import Image as PIL
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame, Paragraph, Table, TableStyle, Image, PageBreak, FrameBreak, NextPageTemplate, Spacer
ROOT=Path(__file__).resolve().parent
for name,file in [('Body','DejaVuSans.ttf'),('Bold','DejaVuSans-Bold.ttf')]:pdfmetrics.registerFont(TTFont(name,str(ROOT/'assets'/file)))
BLUE=colors.HexColor('#1B396A')
body=ParagraphStyle('body',fontName='Body',fontSize=8.6,leading=11.5,spaceAfter=6)
heading=ParagraphStyle('heading',fontName='Bold',fontSize=10.2,leading=13,spaceBefore=7,spaceAfter=6,textColor=BLUE)
small=ParagraphStyle('small',fontName='Body',fontSize=7.1,leading=9.2,spaceAfter=4)
caption=ParagraphStyle('cap',fontName='Body',fontSize=8.3,leading=11.5,spaceAfter=8)
def P(t,style=body):return Paragraph(t,style)
def H(t):return P(t,heading)
def read(n):
 with (ROOT/'results'/n).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
summary=read('summary.csv');stats=read('descriptive_statistics.csv');by={r['model']:r for r in summary};st={(r['model'],r['metric']):r for r in stats}
def val(m,k):return float(by[m][k])
def fmt(m,k):return f'{val(m,k):.3f}'
def abbr(m):return m.replace('Mínima distancia','Mín. dist.').replace('Distances-CS-KNN','Dist.-CS').replace('Direct-CS-KNN','Direct-CS').replace('One-Step adaptado','One-Step*')
def tab(headers,rows,widths):
 data=[[P(escape(str(x)),small) for x in r] for r in [headers]+rows]
 t=Table(data,colWidths=widths,repeatRows=1)
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E8EFF7')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),3),('RIGHTPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),5)]));return t

def fig(name,width=514):
 p=ROOT/'figures'/name;im=PIL.open(p);return Image(str(p),width=width,height=width*im.height/im.width)
def page(c,d):
 c.setFont('Body',7);c.setFillColor(BLUE);c.drawString(38,25,'Héctor Eduardo Gómez Pérez | Sonar UCI | Semilla 42');c.drawRightString(574,25,str(d.page))
 if d.page==1:
  c.setFont('Bold',13);c.drawCentredString(306,753,'KNN, costo de error y reconstrucción matricial en Sonar')
  c.setFont('Body',8);c.drawCentredString(306,735,'Héctor Eduardo Gómez Pérez · Reconocimiento de Patrones · Asignación 5')
  c.drawCentredString(306,721,'Reporte actualizado: matrices de confusión, ROC/AUC y variación entre pliegues')
first=[Frame(38,45,260,654,id='f1'),Frame(314,45,260,654,id='f2')]
cols=[Frame(38,45,260,708,id='c1'),Frame(314,45,260,708,id='c2')]
full=[Frame(38,45,536,708,id='wide')]
doc=BaseDocTemplate(str(ROOT/'Reporte_KNN_HEGP_Visual.pdf'),pagesize=(612,792),title='KNN Sonar: evaluación visual y reproducible',author='Héctor Eduardo Gómez Pérez')
doc.addPageTemplates([PageTemplate(id='first',frames=first,onPage=page,autoNextPageTemplate='columns'),PageTemplate(id='columns',frames=cols,onPage=page),PageTemplate(id='figures',frames=full,onPage=page)])
story=[H('Resumen'),P('Se compararon ocho clasificadores y dos controles constantes en 208 ecos Sonar de UCI. El entrenamiento se redujo a aproximadamente 90% rocas y 10% minas, con ruido en 5% de los registros. Se usaron diez pliegues estratificados, escalado y PCA ajustados en entrenamiento. Direct-CS obtuvo BA='+fmt('Direct-CS-KNN','balanced_accuracy')+' y costo medio de '+f'{val("Direct-CS-KNN","tmc"):.1f}'+'. One-Step adaptado obtuvo AUC-ROC medio='+fmt('One-Step adaptado','roc_auc')+', pero detectó solo 3 de 111 minas con el umbral usado. Las matrices y curvas muestran que discriminación del puntaje y decisión final deben evaluarse por separado. Los resultados pertenecen a las adaptaciones descritas, no a una réplica completa de los artículos.'),
 H('1. Propósito, antecedentes e hipótesis'),P('El propósito es medir cómo las reglas de costo y reconstrucción cambian las minas omitidas, las falsas alarmas y el tiempo. Qin et al. [2] estudian KNN sensible al costo. Zhang et al. [3] aprenden pesos de reconstrucción con penalizaciones; Zhang y Li [4] integran ponderación y grupos. Hang et al. [5] estudian submuestreo en ensambles KNN, citado aquí como contexto, sin ejecutar su ensamble.'),
 P('H1: si se penaliza más omitir una mina, Direct-CS tendrá menor costo que KNN k=5 [2]. H2: con entrenamiento alterado, CM tendrá mayor exactitud balanceada (BA) que KNN k=5, motivado por [3]. H3: si PCA conserva al menos 85% de varianza, el cambio de BA diferirá entre Naive Bayes y KNN. H3 es exploratoria y responde a la consigna. No se atribuyen estas hipótesis como resultados demostrados por los autores.'),
 H('2. Datos y protocolo'),P('Sonar contiene 111 ecos de cilindros metálicos (M=1) y 97 de rocas (R=0), con 60 atributos [1]. En cada pliegue externo se reservan 20 o 21 registros. Solo el entrenamiento se submuestrea: unas 87-88 rocas y 10 minas. Se alteran cinco registros con ruido gaussiano de desviación 2.5 veces la desviación de cada atributo. La prueba conserva aproximadamente 53% de minas.'),
 P('StandardScaler y PCA se ajustan solo con entrenamiento [6]. PCA conserva 15-17 componentes según el pliegue. CV-KNN selecciona k entre 1, 3, 5 y 7 mediante tres pliegues internos, reajustando escalado y PCA en cada uno. Se usa semilla 42. Cada registro aparece una vez en prueba por método. No se ajustaron costos o umbrales con las etiquetas externas.'),FrameBreak(),
 H('3. Métodos y alcance de reproducción'),P('Se ejecutan mínima distancia a centroides, Gaussian Naive Bayes, KNN k=5, CV-KNN, Direct-CS, Distances-CS, CM y One-Step adaptado. Los controles siempre roca y siempre mina permiten detectar ventajas aparentes de predecir una sola clase.'),
 P('Con pérdida 0-1, priors iguales y gaussianas con covarianza común esférica, la regla de Bayes coincide con el centroide euclidiano más próximo. Este supuesto no se prueba como modelo generativo de Sonar.'),
 P('Direct-CS usa p=(vM+2πM)/7, donde vM cuenta minas entre cinco vecinos y πM es el prior de entrenamiento. Los riesgos de decidir mina o roca son 1-p y 10p: el umbral es 1/11. Distances-CS divide las distancias a minas por [10(nR/nM)]^0.3. Son decisiones explícitas del código; el resumen de [2] no permite certificar que reproduzcan todas las fórmulas originales.'),
 P('CM minimiza reconstrucción cuadrática, L1, L2,1 y un término laplaciano de características [3]. El código resuelve por IRLS estabilizado, con hasta 12 iteraciones, y vota usando pesos positivos. One-Step fija Θ y emplea gradiente proyectado con penalizaciones de normas mixtas, sin optimizar los grupos del original [4]. Ambos recalculan W con los atributos del lote de prueba, sin utilizar sus etiquetas. No se demuestra inferencia O(1).'),
 H('4. Evaluación y estadística descriptiva'),P('Las matrices suman las 208 decisiones externas por método. Se muestran conteos y porcentajes dentro de cada clase real. BA promedia sensibilidad y especificidad. G-Mean es su media geométrica. F1 combina precisión y sensibilidad; F2 da más peso a sensibilidad. Costo total=10·FN+FP; el resumen reporta su media por pliegue.'),
 P('ROC recorre umbrales y compara TPR con FPR. AUC-ROC se calcula con puntajes continuos dentro de cada pliegue [7]. Se reporta la media aritmética y DE muestral (ddof=1) de diez AUC. La curva media interpola TPR en 501 valores de FPR. El AUC medio no es el área de esa curva interpolada ni el AUC de puntajes agrupados entre modelos distintos.'),
 P('Las curvas PR se muestran por pliegue y su leyenda usa AP. AP y PR-AUC trapezoidal son medidas diferentes [7]. Un score constante da AUC-ROC=0.5 y AP igual a prevalencia, aunque su PR-AUC trapezoidal sea aproximadamente 0.767 por interpolación. Las barras de DE y las diferencias pareadas describen variación, no son intervalos de confianza ni pruebas de significancia: los entrenamientos de los pliegues se superponen.'),PageBreak(),
 H('5. Resultados y errores'),
 tab(['Modelo','BA','ROC-AUC','DE AUC','TMC'],[[abbr(r['model']),f'{float(r["balanced_accuracy"]):.3f}',f'{float(r["roc_auc"]):.3f}',f'{float(st[r["model"],"roc_auc"]["std"]):.3f}',f'{float(r["tmc"]):.1f}'] for r in summary],[79,40,46,46,43]),
 P('* One-Step es adaptación parcial. BA, ROC-AUC y TMC: medias de diez pliegues. Tabla completa de ocho métricas: figura 8 y summary.csv.',small),
 P('Direct-CS detecta 77 minas y omite 34, con 20 falsas alarmas. KNN detecta 11 y omite 100, sin falsas alarmas. Ambos tienen AUC medio 0.757: sus puntajes mantienen el mismo orden dentro de cada pliegue, pero sus umbrales dan decisiones diferentes. Distances-CS detecta todas las minas, a costa de 95 falsas alarmas sobre 97 rocas; se aproxima al control siempre mina.'),
 P('CM obtiene AUC medio 0.802 y BA 0.616. One-Step adaptado alcanza AUC 0.838 y BA 0.513: su puntaje discrimina mejor que su decisión a umbral 0.5. Esto motiva estudiar calibración o selección de umbral en validación interna; no se ha realizado tal ajuste.'),
 H('6. PCA, tiempos y errores compartidos'),P('PCA reduce BA de Naive Bayes de 62.52% a 59.25% (-3.27 puntos), y de KNN de 55.76% a 54.85% (-0.91 puntos). No se estudió aisladamente el beneficio de estandarizar. La gráfica pareada muestra cada partición antes y después de PCA.'),
 P('Se mide model.fit después del preprocesamiento y la inferencia del lote dividida por sus muestras. Se excluyen escalado, PCA y búsqueda de k. Los tiempos registrados son de esta ejecución y varían entre equipos. CM calcula reconstrucción durante la inferencia.'),
 P('No hay ejemplos fallados por los ocho clasificadores simultáneamente. Con umbral exploratorio de cinco errores, aparecen 65 casos difíciles, todos minas. Las distancias descriptivas se calcularon después de predecir y no se usaron para entrenar. No prueban que esos ejemplos sean ruido.'),FrameBreak(),
 H('7. Discusión y límites'),P('H1 y H2 son compatibles con las medias observadas de costo y BA, respectivamente. H3 presenta cambios de distinta magnitud. Estas observaciones no establecen significancia ni causalidad. En particular, el efecto individual de las penalizaciones de CM no se aisló mediante ablación.'),
 P('Direct-CS tiene la mayor BA media entre estas configuraciones. Distances-CS tiene menor costo, pero casi siempre señala mina. One-Step parcial no permite refutar el método publicado. No se comparan numéricamente nuestras cifras con las tablas de [2-4], porque sus conjuntos y protocolos difieren. Faltan nuevas semillas, prevalencias externas, calibración y una implementación completa de One-Step.'),
 H('8. Reproducción y validación'),P('En Windows/VS Code: crear .venv con Python 3.11, instalar requirements.txt y ejecutar .venv/Scripts/python.exe run_experiment.py. El comando genera CSV, PNG/PDF de las figuras y este reporte. --from-predictions reutiliza los puntajes guardados. La presentación selecciona figuras de figures/ y conserva el método y la evidencia en notas.'),
 P('validate_results.py comprueba etiquetas, pliegues, métricas y conteos. validate_diagnostics.py verifica cada AUC mediante comparación de todos los pares mina/roca (los empates cuentan 1/2), integra la curva ROC y compara los valores, revisa las matrices y las DE. No certifica fidelidad de las adaptaciones a los artículos. La actualización conservó exactamente las predicciones anteriores.'),
 H('Referencias'),
 P('[1] UCI. Connectionist Bench (Sonar, Mines vs. Rocks). doi:10.24432/C5T01Q.',small),
 P('[2] Z. Qin, A. T. Wang, C. Zhang y S. Zhang. Cost-Sensitive Classification with k-Nearest Neighbors. En M. Wang (Ed.), KSEM, LNCS 8041, pp.112-131, 2013. doi:10.1007/978-3-642-39787-5_10. Resumen: opus.lib.uts.edu.au/handle/10453/33945.',small),
 P('[3] S. Zhang, X. Li, M. Zong, X. Zhu y D. Cheng. Learning k for kNN Classification. ACM TIST, 8(3), art.43, 2017. doi:10.1145/2990508.',small),
 P('[4] S. Zhang y J. Li. KNN Classification with One-step Computation. IEEE TKDE, 35(3), pp.2711-2723, 2023. doi:10.1109/TKDE.2021.3119140. Texto: arxiv.org/abs/2012.06047.',small),
 P('[5] H. Hang, Y. Cai, H. Yang y Z. Lin. Under-bagging Nearest Neighbors for Imbalanced Classification, 2021. arxiv.org/abs/2109.00531.',small),
 P('[6] scikit-learn. Common pitfalls. scikit-learn.org/stable/common_pitfalls.html.',small),
 P('[7] scikit-learn. ROC with cross validation: scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html. Metrics and scoring: scikit-learn.org/stable/modules/model_evaluation.html. Consulta: 25-09-2026.',small),
 NextPageTemplate('figures'),PageBreak()]

def graphic_page(title,items):
 story.append(H(title))
 for name,cap,width in items:story.extend([fig(name,width),P(cap,caption)])
 story.append(PageBreak())
graphic_page('Figura 1. Matrices de las referencias y controles',[
 ('confusion_referencias.png','Filas: clase real. Columnas: predicción. Cada celda indica conteo y porcentaje dentro de la clase real. Son las mismas 208 pruebas externas por método.',485),
 ('confusion_controles.png','Controles constantes: referencia para distinguir una clasificación útil de una decisión de una sola clase.',485)])
graphic_page('Figura 2. Matrices de las variantes',[
 ('confusion_variantes.png','Direct-CS reduce omisiones frente a KNN con más falsas alarmas. Distances-CS se aproxima a siempre mina. One-Step parcial omite 108 de 111 minas con el umbral evaluado.',514),
 ('confusion_knn_direct.png','Detalle de la comparación KNN/Direct-CS. La diferencia corresponde a la decisión, no a un aumento de AUC.',470)])
graphic_page('Figura 3. Curvas ROC comparadas',[
 ('roc_referencias.png','Cuatro referencias. Curva media interpolada; AUC de la leyenda es media de AUC por pliegue. La diagonal corresponde al control constante.',495),
 ('roc_variantes.png','Cuatro variantes. La DE de AUC cuantifica dispersión entre pliegues; no es un intervalo de confianza.',495)])
graphic_page('Figura 4. AUC y variabilidad de BA',[
 ('auc_comparacion.png','Cada punto representa un pliegue. Diamante rojo: media. Barra: una DE muestral. La dependencia entre pliegues impide interpretar estas barras como IC.',500),
 ('ba_variabilidad.png','La clasificación a umbral fijo presenta variación considerable. Una media resume el protocolo, no garantiza el mismo rendimiento en datos externos.',500)])
graphic_page('Figura 5. Curvas precisión-sensibilidad',[
 ('pr_referencias.png','Referencias. Cada línea es un pliegue. AP resume precisión promedio y no se identifica con el área trapezoidal PR-AUC.',485),
 ('pr_variantes.png','Variantes. La línea horizontal es la prevalencia global (111/208), solo orientativa: las proporciones de cada pliegue difieren ligeramente.',485)])
graphic_page('Figuras 6 y 7. Comparaciones pareadas y efecto de PCA',[
 ('diferencias_pareadas.png','Figura 6. Cada punto compara una variante con KNN en el mismo pliegue. Diamante negro: media. BA positiva y diferencia de costo negativa favorecen a la variante.',514),
 ('pca_pareado.png','Figura 7. Cada línea conecta el mismo pliegue sin PCA y con PCA. Línea roja: media. Ambos usan escalado, por lo que no se evalúa aquí su efecto separado.',514)])
story.extend([H('Figura 8. Resumen de métricas'),fig('mapa_metricas.png',514),P('Todos los valores son medias de diez pliegues. Acc: exactitud; BA: exactitud balanceada; G: G-Mean. AP y PR-AUC son diferentes. ROC-AUC evalúa orden de puntajes y las métricas de decisión usan los umbrales del código. One-Step* indica adaptación parcial.',caption),H('Archivos de respaldo'),P('results/roc_auc_folds.csv guarda AUC y el punto operativo de cada pliegue. roc_points.csv conserva la curva y sus umbrales. roc_mean_curves.csv contiene la interpolación. confusion_matrices.csv guarda conteos y porcentajes. descriptive_statistics.csv y paired_differences.csv contienen la estadística descriptiva. Todas las figuras están disponibles en PNG y PDF mediante src/diagnostics.py.',caption)])
doc.build(story)
print(ROOT/'Reporte_KNN_HEGP_Visual.pdf')
