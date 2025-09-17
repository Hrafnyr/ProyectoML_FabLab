@echo off
cd /d %~dp0
echo ===============================
echo Instalando dependencias...
echo ===============================

REM Instalar paquetes del requirements.txt
pip install -r requirements.txt

REM Instalar streamlit con PIP
pip install streamlit

REM Se cierra automáticamente al terminar
exit
