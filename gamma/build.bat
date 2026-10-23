@echo off
rem Build gammalog.exe with the Visual Studio C++ toolchain and Geant4.
rem   build.bat [Geant4 install dir]     (default: C:\Program Files\Geant4-11.4)
setlocal
set "G4DIR=%~1"
if "%G4DIR%"=="" set "G4DIR=C:\Program Files\Geant4-11.4"
for /f "usebackq delims=" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
if "%VSDIR%"=="" (echo Visual Studio with C++ tools not found & exit /b 1)
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cmake -S "%~dp0." -B "%~dp0build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGeant4_DIR="%G4DIR%\lib\cmake\Geant4" || exit /b 1
cmake --build "%~dp0build" || exit /b 1
echo Built %~dp0build\gammalog.exe
