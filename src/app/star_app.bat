@echo off
cd /d %~dp0
start "" python -m streamlit run app.py --browser.serverAddress localhost --browser.serverPort 8501
exit
