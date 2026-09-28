# Asignación 5 · KNN avanzado · Sonar original

**Autor:** Héctor Eduardo Gómez Pérez. El archivo `data/sonar.all-data` procede de
[UCI Sonar](https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks)
(208 filas, 60 atributos; SHA-256 en `results/metadata.json`). No se sustituye la colección por
datos sintéticos ni se usan etiquetas de prueba para ajustar los modelos.

## Ejecutar en VS Code (Windows)

Abra esta carpeta con VS Code, abra la terminal y ejecute:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe run_experiment.py
```

La activación de PowerShell no es necesaria. También funciona con Python 3.10+:
`python -m venv .venv`. El experimento usa 90:10 en **entrenamiento**; para
probar 95:5 use `python src/pipeline.py --minority-ratio 0.05` (sobrescribe
`results/`, así que copie previamente los resultados de 90:10).

## Contenido

- `src/pipeline.py`: diez folds externos, tres internos para CV-KNN, ocho
  clasificadores y dos controles; tablas de datos reales.
- `src/cs_knn.py`: reglas de costo implementadas explícitamente.
- `src/reconstruction_knn.py`: CM-kNN con IRLS y adaptación de One-Step con
  Θ fija. La optimización de Θ y grupos del artículo no está reproducida.
- `src/visualization.py`: todas las métricas, costo/latencia, ablación PCA y
  fronteras de los ocho métodos en dos componentes.
- `src/hard_cases.py`: diagnóstico posterior de ejemplos fallados por cinco o
  más métodos; distancias no usadas para ajustar los clasificadores.
- `results/`: resultados completos por fold, predicción y resumen; resultados
  reproducibles salvo los tiempos, que dependen de hardware y carga del equipo.
- `validate_results.py`: verifica las 208 observaciones por modelo y recalcula
  matrices de confusión, métricas, TMC y promedios desde predicciones guardadas.

**Interpretación:** La prueba conserva la prevalencia original (~53% minas),
porque la consigna solicita submuestreo del entrenamiento. Por ello el resultado
no estima directamente una misión AUV con prevalencia 5%. El costo 10:1 y el
ruido 5% son escenarios académicos. PR-AUC es el área trapezoidal; `average_precision`
se registra por separado.

## Nuevos diagnósticos y reporte automático

`run_experiment.py` ejecuta en orden: experimento, gráficos generales, casos difíciles,
matrices/ROC/estadística, dos validadores y reporte PDF de nueve páginas.
Para recrear figuras y reporte con las predicciones incluidas:

```powershell
.venv\Scripts\python.exe run_experiment.py --from-predictions
```

- `src/diagnostics.py`: matrices de los diez métodos, ROC comparativas, curvas PR,
  AUC por pliegue, dispersión, diferencias pareadas y mapa de métricas.
- `figures/`: imágenes PNG listas para insertar en diapositivas y PDF vectoriales.
- `results/roc_auc_folds.csv`: los 100 valores de AUC; `roc_points.csv`: curvas exactas.
- `results/descriptive_statistics.csv`: media, desviación estándar, mediana, cuartiles y extremos.
- `validate_diagnostics.py`: contrasta AUC mediante pares positivo-negativo e integral ROC.
- `build_report.py`: genera `Reporte_KNN_HEGP_Visual.pdf` con las figuras actuales.
- `Presentacion_KNN_HEGP_Visual.pptx`: presentación editable con notas explicativas.

Las figuras y el reporte se regeneran automáticamente. El PowerPoint entregado es
una composición revisada; para nuevas ejecuciones se reemplazan sus imágenes por
los PNG del mismo nombre. No se actualiza automáticamente al ejecutar Python.

Mina (M=1) es la clase positiva. Las matrices suman 208 predicciones externas por
método y muestran porcentajes por clase real. AUC es la media de los diez AUC
individuales; no se mezclan puntajes de modelos entrenados en diferentes pliegues.
Las barras representan desviación estándar muestral, no intervalos de confianza
ni pruebas de significancia. Los pliegues comparten datos de entrenamiento.
Los controles constantes tienen AUC=0.5. AP y área PR trapezoidal son distintas.
Los supuestos y las adaptaciones están documentados en el reporte y en el código.

`requirements-tested.txt` registra las versiones de la ejecución entregada.
