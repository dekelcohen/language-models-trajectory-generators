REM create shortcut: cmd.exe /k "E:\Robotics\VLM_Robotics\language-models-trajectory-generators\setup_env\setup_everytime.bat"

REM Try common install locations
if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\miniconda3\Scripts\activate.bat" vlm_traj
    goto done
)

if exist "%LOCALAPPDATA%\miniconda3\Scripts\activate.bat" (
    call "%LOCALAPPDATA%\miniconda3\Scripts\activate.bat" vlm_traj
    goto done
)

REM Work PC
if exist "C:\ProgramData\miniconda3\Scripts\activate.bat" (
    call "C:\ProgramData\miniconda3\Scripts\activate.bat" vlm_traj
    goto done
)


echo ERROR: Could not find a Conda installation.
echo Tried USERPROFILE, LOCALAPPDATA
exit /b 1

:done
echo Activated vlm_traj


@echo off

REM --- Detect ROBOTICS_HOME ---
if exist "E:\Robotics\" (
    set ROBOTICS_HOME=E:\Robotics
) else if exist "D:\NLP\Robotics\" (
    set ROBOTICS_HOME=D:\NLP\Robotics
) else (
    echo ERROR: Could not find Robotics folder.
    echo Checked:
    echo   E:\Robotics
    echo   D:\NLP\Robotics
    echo.
    echo Please make sure one of these paths exists.
    pause
    exit /b 1
)

echo Using ROBOTICS_HOME=%ROBOTICS_HOME%

REM --- Navigate to project ---
cd /d "%ROBOTICS_HOME%\VLM_Robotics\language-models-trajectory-generators"

REM --- robotic_perception repo py path ---
set PYTHONPATH=%ROBOTICS_HOME%\VLM_Robotics\robotic_perception;%ROBOTICS_HOME%\VLM_Robotics\language-models-trajectory-generators

REM --- Activate conda environment ---
call conda activate vlm_traj

REM --- Optional git pull ---
echo.
set /p gitchoice=Do you want to run git pull? (y/n): 

if /i "%gitchoice%"=="y" (
    git pull
)

echo.
echo Environment ready.
