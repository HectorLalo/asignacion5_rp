"""Ejecuta el experimento y genera tablas, figuras y reporte PDF.

python run_experiment.py
python run_experiment.py --from-predictions  # reutiliza las predicciones guardadas
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-predictions',action='store_true',help='Recrear diagnosticos y reporte sin reentrenar')
    args=parser.parse_args()
    steps=[] if args.from_predictions else ['src/pipeline.py']
    steps+=['src/visualization.py','src/hard_cases.py','src/diagnostics.py','validate_results.py','validate_diagnostics.py','build_report.py']
    for script in steps:
        print(f'\nEjecutando {script}',flush=True)
        subprocess.run([sys.executable,str(ROOT/script)],cwd=ROOT,check=True)
    print('\nCOMPLETADO: results/, figures/ y Reporte_KNN_HEGP_Visual.pdf')

if __name__=='__main__':main()
