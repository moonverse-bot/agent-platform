@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo.
echo   智能工单 / 客服 Agent 平台
echo   ------------------------------
echo.

if not exist ".venv\Scripts\python.exe" goto :setup
echo [1/3] 检测到虚拟环境,跳过创建与安装。
goto :check_env

:setup
echo [1/3] 首次运行:创建虚拟环境...
python -m venv .venv
if errorlevel 1 goto :fail
echo [2/3] 安装依赖(第一次会慢一些)...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :fail

:check_env
if exist ".env" goto :run
echo [3/3] 未找到 .env,已从 .env.example 复制一份。
copy /y ".env.example" ".env" >nul
echo.
echo   请打开 .env,把 DEEPSEEK_API_KEY 换成你自己的 Key,
echo   保存后重新运行本脚本。
echo.
pause
exit /b 0

:run
findstr /c:"sk-xxxxxxxxxxxxxxxx" ".env" >nul
if not errorlevel 1 echo   [!] .env 里的 DEEPSEEK_API_KEY 似乎还是模板值,AI 对话可能失败。
echo.
echo   正在启动:http://127.0.0.1:8000
echo   按 Ctrl+C 停止服务。
echo.
start "" http://127.0.0.1:8000
".venv\Scripts\python.exe" -m uvicorn app.main:app --port 8000
exit /b 0

:fail
echo.
echo   [x] 启动失败,请把上面的报错信息发给我。
pause
exit /b 1
