@echo off
cd /d %~dp0
echo ===============================
echo Instalando dependencias...
echo ===============================

REM Instalar paquetes del requirements.txt
pip install -r requirements.txt

REM Se cierra automáticamente al terminar
exit
