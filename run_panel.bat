@echo off
REM Cross-model panel on the laptop under the 2026-10-23 design (6 learners, 8192-token context):
REM Llama 3.1 8B, Gemma 3 12B and Granite 4.2 8B, reduced design (A, B, C, D, D_minus, E, E_minus;
REM 24 main tasks; 3 seeds). The 9B and 27B main runs and the large models run on ISCA.
REM Before starting: Ollama > Settings > Context length = 8k (or larger), then restart Ollama.
REM Safe to stop (close this window) and start again: it continues where it stopped.
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe
set LOG=data\jobs\run_panel.log
if not exist data\jobs mkdir data\jobs
echo ==== started %date% %time% ==== >> %LOG%
%PY% -m pip install -q matplotlib >> %LOG% 2>&1

for %%M in (llama31_8b gemma3_12b granite42_8b) do (
  echo [%%M] references %date% %time% >> %LOG%
  %PY% scripts\build_reference.py --tasks all --n 100 --profile %%M >> %LOG% 2>&1
  echo [%%M] sessions %date% %time% >> %LOG%
  %PY% scripts\run_experiment.py --tasks main --conditions A,B,C,D,D_minus,E,E_minus --seeds 1-3 --profile %%M >> %LOG% 2>&1
)

echo [analysis] %date% %time% >> %LOG%
%PY% scripts\analyze.py --all-models >> %LOG% 2>&1
%PY% scripts\model_consistency.py >> %LOG% 2>&1

echo ==== finished %date% %time% ==== >> %LOG%
echo Panel finished. See data\jobs\run_panel.log
pause
