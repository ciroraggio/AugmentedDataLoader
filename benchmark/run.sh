#!/bin/bash

total_files=2
completed_files=0

echo "Esecuzione cpu_bench.py"
python cpu_bench.py > cpu_bench_log.txt
if [ $? -eq -1 ]; then
    echo "Errore durante l'esecuzione di cpu_bench.py"
    exit 1
fi
echo

completed_files=$((completed_files + 1))
DisplayProgress

echo "Esecuzione gpu_bench.py"
python gpu_bench.py > gpu_bench_log.txt
if [ $? -eq -1 ]; then
    echo "Errore durante l'esecuzione di gpu_bench.py"
    exit 1
fi
echo

completed_files=$((completed_files + 1))
DisplayProgress

echo "Esecuzione completata."
read -p "Premi INVIO per uscire."

exit 0

DisplayProgress() {
    progress=$((completed_files * 100 / total_files))
    echo "Avanzamento: $completed_files/$total_files"
}
