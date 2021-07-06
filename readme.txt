
======================== INSTALL PACKAGES =======================
Open command prompt 
CD to the Artificial Intelligence folder
Then install all required packages with the following command:
pip install -r requirements.txt 

WARNING:
PYTHON 3.7.3 IS REQUIRED. 

======================== GPU TRAINING ===========================
CUDA 10.0
CuDNN V7.6.5
To gain access to these obsolete versions membership of nvidia development is needed this can be requested at:
https://developer.nvidia.com/rdp/cudnn-download

WARNING:
MAKE SURE YOUR GPU IS COMPATIBLE TO USE CUDA

======================== TESSERACT ON WINDOWS ===================
In the Artificial Intelligence folder there's another folder
called 'Tesseract-Installation' 

1. Navigate to this folder
2. Install the tesseract-ocr-w64-setup-v5.0.0-alpha.20201127.exe
in C:\Program Files (x86)\Tesseract- OCR
3. To access Tesseract-OCR from any location you may have to add
the installation directory to the Path Variables.
4. Move the .traineddata files in 'Tesseract-Installation to
C:\Program Files (x86)\Tesseract-OCR\tessdata
5. Then do 'pip install pytesseract == 0.3.7' 
6. Open script Detection.py
7. At the bottom of the imports you should see comments
with further instruction. 

For more information (and if you're interested installing this
for Linux go to the following link:
https://github.com/tesseract-ocr/tesseract


====================== TEST AI ==================================
Once you've installed Tesseract + everything else in 
requirements.txt you can test if the installation succeeded
by running the 'Demo.ipynb' Notebook. If this runs with out
errors then your installation is a success! 