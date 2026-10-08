@echo off
setlocal
if defined NCA_PYTHON goto run
set "NCA_PYTHON=C:\LAB\ai-aec-playground\PROJECTS\Constraint-Based-Architectural-NCA\.venv\Scripts\python.exe"
if exist "%NCA_PYTHON%" goto run
echo Set NCA_PYTHON to the tested Python environment described in START-HERE.md.
pause
exit /b 1
:run
"%NCA_PYTHON%" "%~dp0launch.py" %*
if errorlevel 1 pause
