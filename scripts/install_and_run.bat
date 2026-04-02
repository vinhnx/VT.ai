@echo off
REM VT.ai All-in-One Installer and Runner for Windows
REM This script installs all dependencies and runs VT.ai in one command

setlocal enabledelayedexpansion

:: Colors (Windows 10+)
for /F "tokens=1,2 delims=#" %%a in ('"prompt #$H#$E# & echo on & for %%b in (1) do rem"') do (
  set "DEL=%%a"
  set "COLOR=%%b"
)

set "HEADER=[VT.ai Installer]"
set "SUCCESS=✓"
set "WARNING=⚠"
set "ERROR=✗"

echo ========================================
echo %HEADER% VT.ai All-in-One Installer
echo ========================================
echo.
echo This script will:
echo   1. Check Python version (requires 3.11)
echo   2. Install uv package manager (if needed)
echo   3. Create a virtual environment
echo   4. Install VT.ai and all dependencies
echo   5. Configure API keys
echo   6. Run VT.ai
echo.

set /p CONTINUE="Continue? (y/n) "
if /i not "%CONTINUE%"=="y" (
    echo Installation cancelled.
    exit /b 0
)

echo.

:: Check Python version
echo Checking Python version...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo %ERROR% Python not found!
    echo Please install Python 3.11 from https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Found Python %PYTHON_VERSION%

:: Check if uv is installed
echo.
echo Checking uv package manager...
where uv >nul 2>&1
if %errorlevel% equ 0 (
    echo %SUCCESS% uv is already installed
    set "UV_CMD=uv"
) else (
    echo %WARNING% uv not found, installing...
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    
    :: Refresh PATH
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
    
    where uv >nul 2>&1
    if %errorlevel% equ 0 (
        echo %SUCCESS% uv installed successfully
        set "UV_CMD=uv"
    ) else (
        echo %WARNING% uv installation failed, falling back to pip
        set "UV_CMD=pip"
    )
)

:: Create virtual environment
echo.
echo Setting up virtual environment...
if exist ".venv" (
    echo %WARNING% Virtual environment already exists
    set /p RECREATE="Do you want to recreate it? (y/n) "
    if /i "!RECREATE!"=="y" (
        rmdir /s /q .venv
        echo %SUCCESS% Removed old virtual environment
    ) else (
        echo %SUCCESS% Using existing virtual environment
    )
)

if not exist ".venv" (
    echo Creating virtual environment...
    if "!UV_CMD!"=="uv" (
        uv venv --python python3.11
    ) else (
        python -m venv .venv
    )
    echo %SUCCESS% Virtual environment created
)

:: Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo %SUCCESS% Virtual environment activated

:: Install dependencies
echo.
echo Installing Dependencies...
echo Installing VT.ai and all dependencies (this may take a few minutes)...

if "!UV_CMD!"=="uv" (
    uv pip install -e ".[dev]"
) else (
    pip install -e ".[dev]"
)

echo %SUCCESS% All dependencies installed

:: Configure API keys
echo.
echo Configuring API Keys...

set "CONFIG_DIR=%USERPROFILE%\.config\vtai"
set "ENV_FILE=%CONFIG_DIR%\.env"

if not exist "%CONFIG_DIR%" mkdir "%CONFIG_DIR%"

if exist "%ENV_FILE%" (
    echo %WARNING% API configuration already exists at %ENV_FILE%
    set /p UPDATE="Do you want to update it? (y/n) "
    if /i not "!UPDATE!"=="y" (
        echo %SUCCESS% Using existing configuration
        goto :RUN_PROMPT
    )
)

:: Create new .env file
echo # VT.ai API Configuration > "%ENV_FILE%"
echo # Created on %DATE% >> "%ENV_FILE%"
echo. >> "%ENV_FILE%"

echo Enter your API keys (press Enter to skip):
echo.

set /p OPENAI_KEY="OpenAI API Key: "
if not "!OPENAI_KEY!"=="" (
    echo OPENAI_API_KEY='!OPENAI_KEY!' >> "%ENV_FILE%"
    echo %SUCCESS% OpenAI API key saved
)

set /p ANTHROPIC_KEY="Anthropic API Key: "
if not "!ANTHROPIC_KEY!"=="" (
    echo ANTHROPIC_API_KEY='!ANTHROPIC_KEY!' >> "%ENV_FILE%"
    echo %SUCCESS% Anthropic API key saved
)

set /p GEMINI_KEY="Google Gemini API Key: "
if not "!GEMINI_KEY!"=="" (
    echo GEMINI_API_KEY='!GEMINI_KEY!' >> "%ENV_FILE%"
    echo %SUCCESS% Google Gemini API key saved
)

set /p TAVILY_KEY="Tavily API Key (for web search): "
if not "!TAVILY_KEY!"=="" (
    echo TAVILY_API_KEY='!TAVILY_KEY!' >> "%ENV_FILE%"
    echo %SUCCESS% Tavily API key saved
)

echo.
echo %SUCCESS% API keys saved to %ENV_FILE%
echo.
echo You can always update these later by editing: %ENV_FILE%

:RUN_PROMPT
echo.
set /p RUN_NOW="Do you want to run VT.ai now? (y/n) "
if /i not "!RUN_NOW!"=="y" (
    echo.
    echo %SUCCESS% Installation complete!
    echo.
    echo To run VT.ai later:
    echo   1. Activate the virtual environment:
    echo      .venv\Scripts\activate
    echo   2. Run VT.ai:
    echo      chainlit run vtai\app
    echo.
    echo Or simply run this script again to start VT.ai!
    pause
    exit /b 0
)

:: Run VT.ai
echo.
echo ========================================
echo Starting VT.ai
echo ========================================
echo.
echo Launching VT.ai...
echo The application will open in your default browser at: http://localhost:8000
echo.
echo Press Ctrl+C to stop the application
echo.

chainlit run vtai\app -w

endlocal
