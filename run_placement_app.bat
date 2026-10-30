@echo off
echo ===============================================================================
echo Starting Placement Analysis & Intelligence System...
echo Lab Alignment: Data Science and Big Data Lab (CO1, CO2, CO3, CO4)
echo ===============================================================================
echo Pre-seeded Logins:
echo   [Admin/Placement Cell]: username 'admin'   / password 'admin123'
echo   [Student]:              username 'pranay'  / password 'student123'
echo ===============================================================================
python -m streamlit run placement_app.py
pause
