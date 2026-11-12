@echo off
title ClearParty: Dual-Track Audio AI (CPU Accelerated)
echo =====================================================================
echo    ClearParty: Deep Speaker Separation & Speech Enhancement
echo    100%% CPU-Optimized Web Application (No GPU Required)
echo =====================================================================
echo.
echo Launching Streamlit Frontend at http://localhost:8501 ...
echo Press Ctrl+C in this terminal to stop the application.
echo.

"C:\Python314\python.exe" -m streamlit run "D:\cocktail_party_project\webapp.py" --server.port 8501 --server.headless false

pause
