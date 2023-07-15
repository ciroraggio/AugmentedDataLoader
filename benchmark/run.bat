@echo off

setlocal enabledelayedexpansion

set total_files=2
set /a completed_files=0

echo Esecuzione cpu_bench.py
python cpu_bench.py > cpu_bench_log.txt
if %errorlevel% equ -1 (
    echo Errore durante l'esecuzione di cpu_bench.py
    exit /b
)
echo.
set /a completed_files+=1
call :DisplayProgress

echo Esecuzione gpu_bench.py
python gpu_bench.py > gpu_bench_log.txt
if %errorlevel% equ -1 (
    echo Errore durante l'esecuzione di gpu_bench.py
    exit /b
)
echo.
set /a completed_files+=1
call :DisplayProgress

echo Esecuzione completata.
pause
exit /b

:DisplayProgress
REM Calcola la percentuale di avanzamento
set /a progress=(completed_files * 100) / total_files

REM Stampa l'indicatore di avanzamento
echo Avanzamento: %completed_files%/%total_files%
exit /b