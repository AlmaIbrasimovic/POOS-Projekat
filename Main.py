# Main.py
# Zakomentarisane linije 118 i 119 (Alma) --- pomocni prozori prilikom obrade (smetaju)

import cv2
import numpy as np
import os
import sys
from PyQt5.QtWidgets import QWidget, QDesktopWidget, QApplication, QPushButton, QFileDialog, QLabel, QHBoxLayout
from PyQt5.QtCore import pyqtSlot, QObject, Qt
from PyQt5.QtGui import QIcon, QPixmap, QPalette
from tkinter import messagebox
import ctypes 

import DetectChars
import DetectPlates
import PossiblePlate

class App(QWidget):
	def __init__(self): #Konstruktor
		super().__init__()
		self.initUI()
		self.putanja = ""

	def getPutanja(self):
		return self.putanja

	def setPutanja(self, putanja):
		self.putanja = putanja

	def initUI(self):
		self.setWindowTitle('Učitavanje slika')
		self.setFixedSize(500,400)

		# Postavke za button A
		button_A = QPushButton('Učitaj sliku registarske tablice',self)
		button_A.setToolTip('Dugme za ucitavanje slike registarskih tablica!')
		button_A.move(145,100)
		button_A.setFixedSize(190,50)

		# Postavke za button B
		button_B = QPushButton('Ugasi aplikaciju',self)
		button_B.setToolTip('Dugme za gašenje aplikacije!')
		button_B.move(145,200)
		button_B.setFixedSize(190,50)
		button_B.clicked.connect(self.close)
	
		button_A.clicked.connect(self.ucitajSliku) # Povezivanje klika sa funkcijom
		self.center() 
		self.show()

	@pyqtSlot() # Funkcija za ucitavanje slike
	def ucitajSliku(self):
		a = self.kreirajDijalog()
		return a
		
	def center(self): # Da bi pozicionirali ne sredinu ekrana
		qr = self.frameGeometry()
		cp = QDesktopWidget().availableGeometry().center()
		qr.moveCenter(cp)
		self.move(qr.topLeft())
	
	def kreirajDijalog(self):
		options = QFileDialog.Options()
		hbox = QHBoxLayout(self)
		files = QFileDialog.getOpenFileName(self, "Učitaj sliku", "", "JPG (*.jpg);;PNG (*.png)")
		self.setPutanja(files[0])
		if files[0] : self.close() # Provjera da li se učitala slika

SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

showSteps = False
def main():

    blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()         # KNN treniranje

    if blnKNNTrainingSuccessful == False:                               # ukoliko KNN nije uspjesno
        print("\nGreska: KNN treniranje nije uspjesno!\n")  
        return                                                        
    # end if

    app = QApplication(sys.argv)
    w = QWidget()
    program = App()
    app.setStyle("Fusion")
    app.exec_()
    imgOriginalScene  = cv2.imread(program.getPutanja())             

    if imgOriginalScene is None:                       
        print("\nGreska: slika nije ucitana!\n\n") 
        os.system("pause")                                  
        return                                           
    # end if

    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)           # detekcija tablica

    listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)        # detekcija charova u tablicama 
    cv2.imshow("imgOriginalScene", imgOriginalScene)         

    if len(listOfPossiblePlates) == 0:                          # ukoliko se nije pronasla niti jedna tablica
        print("\nNije detektovana niti jedna registarska tablica!\n")  
    else:                                                       # else
        listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)
        licPlate = listOfPossiblePlates[0]

        # Pomocni prozori koji nisu potrebni, ali moze se upaliti ako treba
        #cv2.imshow("imgPlate", licPlate.imgPlate)        
        #cv2.imshow("imgThresh", licPlate.imgThresh)

        if len(licPlate.strChars) == 0:                     # iukoliko se nije pronasao niti jedan char na tablici
            print("\nNisu detektovani karakteri!\n\n")  
            return                                         
        # end if

        drawRedRectangleAroundPlate(imgOriginalScene, licPlate)             # crveni pravougaonik

        
        #print("\nRegistarska tablica prepoznata sa slike = " + licPlate.strChars + "\n") 
        print("----------------------------------------")

        writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)        

        cv2.imshow("imgOriginalScene", imgOriginalScene)              

        cv2.imwrite("imgOriginalScene.png", imgOriginalScene)           
        ctypes.windll.user32.MessageBoxW(0, "Registarska tablica prepoznata sa slike = " + licPlate.strChars, "Tablica")

    # end if else

    cv2.waitKey(0)			

    return
# end main

###################################################################################################
def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):

    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)           

    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)      
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)
# end function

###################################################################################################
def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    ptCenterOfTextAreaX = 0                          
    ptCenterOfTextAreaY = 0

    ptLowerLeftTextOriginX = 0                       
    ptLowerLeftTextOriginY = 0

    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX                      
    fltFontScale = float(plateHeight) / 30.0                  
    intFontThickness = int(round(fltFontScale * 1.5))        

    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale, intFontThickness)     
    ( (intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg ) = licPlate.rrLocationOfPlateInScene

    intPlateCenterX = int(intPlateCenterX)             
    intPlateCenterY = int(intPlateCenterY)

    ptCenterOfTextAreaX = int(intPlateCenterX)   

    if intPlateCenterY < (sceneHeight * 0.75):                                                 
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(round(plateHeight * 1.6))      
    else:                                                                                       
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(round(plateHeight * 1.6))      
    # end if

    textSizeWidth, textSizeHeight = textSize            
    ptLowerLeftTextOriginX = int(ptCenterOfTextAreaX - (textSizeWidth / 2))          
    ptLowerLeftTextOriginY = int(ptCenterOfTextAreaY + (textSizeHeight / 2))     
    cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace, fltFontScale, SCALAR_YELLOW, intFontThickness)
# end function

###################################################################################################
if __name__ == "__main__":
    main()


















