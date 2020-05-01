SET PYTHON_ROOT=D:\Program Files\Python

SET PIP="%PYTHON_ROOT%\Scripts\pip.exe"
SET PYTHON="%PYTHON_ROOT%\python.exe"

%PIP% install virtualenv
%PYTHON% -m venv dynamicTBA

call .\Collaborative-Story-Teller-Bot\Scripts\activate.bat
call pip install -r req.txt
deactivate
