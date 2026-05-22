@echo off
echo Installing dependencies...
pip install -q -r requirements.txt
echo.
echo Running agent adversarial evaluation...
if "%ANTHROPIC_API_KEY%"=="" (
    echo ERROR: ANTHROPIC_API_KEY environment variable not set.
    echo Set it with: set ANTHROPIC_API_KEY=sk-ant-...
    exit /b 1
)
python run_eval.py
