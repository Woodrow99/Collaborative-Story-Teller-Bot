set PATH=%PATH%;D:\Program Files\Python\
set PATH=%PATH%;D:\Program Files\Python\Scripts
pip install virtualenv
virtualenv venv
call .\venv\Scripts\activate.bat
call pip install -r requirements.txt
deactivate

