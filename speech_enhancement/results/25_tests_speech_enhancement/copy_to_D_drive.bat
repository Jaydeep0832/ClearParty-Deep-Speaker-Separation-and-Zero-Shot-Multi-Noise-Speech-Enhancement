@echo off
echo ======================================================================
echo Copying 25 Tests Results to D:\cocktail_party_project\Speech_Enhancement\results
echo ======================================================================
robocopy "C:\Users\jayde\Speech_Enhancement\results\25_tests_speech_enhancement" "D:\cocktail_party_project\Speech_Enhancement\results\25_tests_speech_enhancement" /E
echo Done!
pause
