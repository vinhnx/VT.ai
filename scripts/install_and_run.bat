@echo off
REM VT.ai Native Installer for Windows
REM Installs VT.ai with all dependencies using uv
REM Supports: Windows (PowerShell/CMD/WSL)

setlocal enabledelayedexpansion

REM Configuration
set "REPO=vinhnx/VT.ai"
set "PYTHON_VERSION=3.11"
set "VENV_DIR=.venv"
set "CONFIG_DIR=%USERPROFILE%\.config\vtai"
set "ENV_FILE=%CONFIG_DIR%\.env"

REM Colors (Windows 10+)
for /F "tokens=1,2 delims=#" %%a in ('"prompt #$H#$E# & echo on & for %%b in (1) do rem"') do (
  set "DEL=%%a"
  set "COLOR=%%b"
)

set "BLUE=[INFO]"
set "GREEN=✓"
set "YELLOW=⚠"
set "RED=✗"

REM Logging functions
set "SCRIPT_NAME=%~nx0"

:check_if_cloned
if not exist "pyproject.toml" (
    echo %BLUE% VT.ai not found in current directory
    echo %BLUE% Cloning repository...
    git clone "https://github.com/%REPO%.git" vtai-temp
    cd vtai-temp
    set "CLEANUP_NEEDED=1"
)

:check_requirements
echo %BLUE% Checking requirements...
where curl >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED% curl is required for installation
    exit /b 1
)
echo %GREEN% Requirements check passed

:check_python
echo %BLUE% Checking Python version (requires %PYTHON_VERSION%)...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED% Python %PYTHON_VERSION% not found
    echo %BLUE% Please install Python %PYTHON_VERSION% from https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set "PY_VER=%%i"
echo %GREEN% Found Python !PY_VER!

:check_uv
echo %BLUE% Checking uv package manager...
where uv >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN% uv is already installed
    set "UV_CMD=uv"
    goto :create_venv
)

echo %YELLOW% uv not found, installing...
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" >nul 2>&1

REM Refresh PATH
set "PATH=%USERPROFILE%\.local\bin;%PATH%"

where uv >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN% uv installed successfully
    set "UV_CMD=uv"
) else (
    echo %YELLOW% uv installation failed, falling back to pip
    set "UV_CMD=pip"
)

:create_venv
echo %BLUE% Setting up virtual environment...
if exist "%VENV_DIR%" (
    echo %YELLOW% Virtual environment already exists
    set /p RECREATE="Do you want to recreate it? (y/n) "
    if /i "!RECREATE!"=="y" (
        rmdir /s /q "%VENV_DIR%"
        echo %GREEN% Removed old virtual environment
    ) else (
        echo %GREEN% Using existing virtual environment
        goto :install_deps
    )
)

echo %BLUE% Creating virtual environment with Python %PYTHON_VERSION%...
if "!UV_CMD!"=="uv" (
    uv venv --python python%PYTHON_VERSION% >nul 2>&1
) else (
    python -m venv "%VENV_DIR%"
)
echo %GREEN% Virtual environment created

REM Activate virtual environment
echo %BLUE% Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"
echo %GREEN% Virtual environment activated

:install_deps
echo %BLUE% Installing VT.ai and all dependencies...
echo %BLUE% This may take a few minutes depending on your connection

if "!UV_CMD!"=="uv" (
    uv pip install -e ".[dev]" >nul 2>&1
) else (
    pip install -e ".[dev]" >nul 2>&1
)

if %errorlevel% equ 0 (
    echo %GREEN% All dependencies installed
) else (
    echo %RED% Failed to install dependencies
    pause
    exit /b 1
)

:configure_api
echo %BLUE% Configuring API keys...

if not exist "%CONFIG_DIR%" mkdir "%CONFIG_DIR%"

if exist "%ENV_FILE%" (
    echo %YELLOW% API configuration already exists at %ENV_FILE%
    set /p UPDATE="Do you want to update it? (y/n) "
    if /i not "!UPDATE!"=="y" (
        echo %GREEN% Using existing configuration
        goto :run_prompt
    )
)

REM Create new .env file
echo # VT.ai API Configuration > "%ENV_FILE%"
echo # Created on %DATE% >> "%ENV_FILE%"
echo. >> "%ENV_FILE%"

echo.
echo %BLUE% Enter your API keys (press Enter to skip):
echo.

set /p OPENAI_KEY="OpenAI API Key: "
if not "!OPENAI_KEY!"=="" (
    echo OPENAI_API_KEY='!OPENAI_KEY!' >> "%ENV_FILE%"
    echo %GREEN% OpenAI API key saved
)

set /p ANTHROPIC_KEY="Anthropic API Key: "
if not "!ANTHROPIC_KEY!"=="" (
    echo ANTHROPIC_API_KEY='!ANTHROPIC_KEY!' >> "%ENV_FILE%"
    echo %GREEN% Anthropic API key saved
)

set /p GEMINI_KEY="Google Gemini API Key: "
if not "!GEMINI_KEY!"=="" (
    echo GEMINI_API_KEY='!GEMINI_KEY!' >> "%ENV_FILE%"
    echo %GREEN% Google Gemini API key saved
)

set /p TAVILY_KEY="Tavily API Key (for web search): "
if not "!TAVILY_KEY!"=="" (
    echo TAVILY_API_KEY='!TAVILY_KEY!' >> "%ENV_FILE%"
    echo %GREEN% Tavily API key saved
)

echo.
echo %GREEN% API keys saved to %ENV_FILE%
echo %BLUE% You can always update these later by editing: %ENV_FILE%

:run_prompt
echo.
set /p RUN_NOW="Do you want to run VT.ai now? (y/n) "
if /i not "!RUN_NOW!"=="y" (
    goto :install_complete
)

:run_vtai
echo.
echo ========================================
echo %BLUE% Starting VT.ai
echo ========================================
echo.
echo %BLUE% Launching VT.ai...
echo %BLUE% The application will open in your default browser at: http://localhost:8000
echo.
echo %BLUE% Press Ctrl+C to stop the application
echo.

chainlit run vtai\app -w
goto :cleanup

:install_complete
echo.
echo %GREEN% Installation complete!
echo.
echo %BLUE% To run VT.ai later:
echo   1. Activate the virtual environment:
echo      %VENV_DIR%\Scripts\activate
echo   2. Run VT.ai:
echo      chainlit run vtai\app
echo.
echo %BLUE% Or simply run this script again to start VT.ai!

:cleanup
if defined CLEANUP_NEEDED (
    cd ..
    rmdir /s /q vtai-temp
)

endlocal
pause
