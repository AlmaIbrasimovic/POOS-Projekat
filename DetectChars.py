# DetectChars.py
import os

import cv2
import numpy as np
import math
import random

import Main
import Preprocess
import PossibleChar

# module level variables ##########################################################################

kNearest = cv2.ml.KNearest_create()

        # konstante za checkIfPossibleChar, provjerava samo jedan char (ne poredi sa drugim charom)
MIN_PIXEL_WIDTH = 2
MIN_PIXEL_HEIGHT = 8

MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 1.0

MIN_PIXEL_AREA = 80

        # konstante za poredenje dva chara
MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3
MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0

MAX_CHANGE_IN_AREA = 0.5

MAX_CHANGE_IN_WIDTH = 0.8
MAX_CHANGE_IN_HEIGHT = 0.2

MAX_ANGLE_BETWEEN_CHARS = 12.0

        # ostale konstante
MIN_NUMBER_OF_MATCHING_CHARS = 3

RESIZED_CHAR_IMAGE_WIDTH = 20
RESIZED_CHAR_IMAGE_HEIGHT = 30

MIN_CONTOUR_AREA = 100

###################################################################################################
def loadKNNDataAndTrainKNN():
    allContoursWithData = []                # prazne liste
    validContoursWithData = []              

    try:
        npaClassifications = np.loadtxt("classifications.txt", np.float32)                  # ucitati klasifikacije za treniranje
    except:                                                                                 
        print("error, unable to open classifications.txt, exiting program\n") 
        os.system("pause")
        return False                                                                        
    # end try

    try:
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)                 # ucitati treniranje za slike
    except:                                                                                 
        print("error, unable to open flattened_images.txt, exiting program\n")  
        os.system("pause")
        return False                                                                        
    # end try

    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))       # pretvoriti numpy niz u 1d, neohodno za treniranje

    kNearest.setDefaultK(1)                                                             # postaviti pocetni K na 1

    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)           # treniranje knn objekta

    return True               
# end function

###################################################################################################
def detectCharsInPlates(listOfPossiblePlates):
    intPlateCounter = 0
    imgContours = None
    contours = []

    if len(listOfPossiblePlates) == 0:          # ukoliko je lista mogucih tablica prazna
        return listOfPossiblePlates             # return
    # end if

    for possiblePlate in listOfPossiblePlates:          # za svaku mogucu tablicu

        possiblePlate.imgGrayscale, possiblePlate.imgThresh = Preprocess.preprocess(possiblePlate.imgPlate)     # predprocesi da se dobiju grayscale i threshold slike

        if Main.showSteps == True: # prikazi korake 
            cv2.imshow("5a", possiblePlate.imgPlate)
            cv2.imshow("5b", possiblePlate.imgGrayscale)
            cv2.imshow("5c", possiblePlate.imgThresh)
        # end if

                # povecati velicinu slike da bi se bolje vidjelo i lakse detektovali charovi
        possiblePlate.imgThresh = cv2.resize(possiblePlate.imgThresh, (0, 0), fx = 1.6, fy = 1.6)

                # threshold  ponovo da bi se otklonile sive povrsine
        thresholdValue, possiblePlate.imgThresh = cv2.threshold(possiblePlate.imgThresh, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        if Main.showSteps == True: 
            cv2.imshow("5d", possiblePlate.imgThresh)

                # pronalazi sve moguce charove na slici
                # ova funkcija prvo pronalazi sve konture, i onda samo ukljucuje konture koje bi mogle biti charovi
        listOfPossibleCharsInPlate = findPossibleCharsInPlate(possiblePlate.imgGrayscale, possiblePlate.imgThresh)

        if Main.showSteps == True:
            height, width, numChannels = possiblePlate.imgPlate.shape
            imgContours = np.zeros((height, width, 3), np.uint8)
            del contours[:]                                         # ocistiti listu kontura

            for possibleChar in listOfPossibleCharsInPlate:
                contours.append(possibleChar.contour)
            # end for

            cv2.drawContours(imgContours, contours, -1, Main.SCALAR_WHITE)

            cv2.imshow("6", imgContours)
        # end if #

                # nakon sto imamo listu svih mogucih charova, pronaci grupu slicnih charova na slici
        listOfListsOfMatchingCharsInPlate = findListOfListsOfMatchingChars(listOfPossibleCharsInPlate)

        if Main.showSteps == True:
            imgContours = np.zeros((height, width, 3), np.uint8)
            del contours[:]

            for listOfMatchingChars in listOfListsOfMatchingCharsInPlate:
                intRandomBlue = random.randint(0, 255)
                intRandomGreen = random.randint(0, 255)
                intRandomRed = random.randint(0, 255)

                for matchingChar in listOfMatchingChars:
                    contours.append(matchingChar.contour)
                # end for
                cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
            # end for
            cv2.imshow("7", imgContours)
        # end if

        if (len(listOfListsOfMatchingCharsInPlate) == 0):			# ukoliko nisu pronadene grupe slicnih charova

            if Main.showSteps == True: 
                print("chars found in plate number " + str(
                    intPlateCounter) + " = (none), click on any image and press a key to continue . . .")
                intPlateCounter = intPlateCounter + 1
                cv2.destroyWindow("8")
                cv2.destroyWindow("9")
                cv2.destroyWindow("10")
                cv2.waitKey(0)
            # end if

            possiblePlate.strChars = ""
            continue						# vracanje na pocetak for petlje
        # end if

        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):                              
            listOfListsOfMatchingCharsInPlate[i].sort(key = lambda matchingChar: matchingChar.intCenterX)        # sortirati charova s lijeva na desno
            listOfListsOfMatchingCharsInPlate[i] = removeInnerOverlappingChars(listOfListsOfMatchingCharsInPlate[i])              # i ukloniti charove koji se preklapaju
        # end for

        if Main.showSteps == True:
            imgContours = np.zeros((height, width, 3), np.uint8)

            for listOfMatchingChars in listOfListsOfMatchingCharsInPlate:
                intRandomBlue = random.randint(0, 255)
                intRandomGreen = random.randint(0, 255)
                intRandomRed = random.randint(0, 255)

                del contours[:]

                for matchingChar in listOfMatchingChars:
                    contours.append(matchingChar.contour)
                # end for

                cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
            # end for
            cv2.imshow("8", imgContours)
        # end if

        intLenOfLongestListOfChars = 0
        intIndexOfLongestListOfChars = 0

                # proci petljom kroz sve vektore poklapajucih charova i uzeti indeks onog koji se najvise poklapa
        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):
            if len(listOfListsOfMatchingCharsInPlate[i]) > intLenOfLongestListOfChars:
                intLenOfLongestListOfChars = len(listOfListsOfMatchingCharsInPlate[i])
                intIndexOfLongestListOfChars = i
            # end if
        # end for

        longestListOfMatchingCharsInPlate = listOfListsOfMatchingCharsInPlate[intIndexOfLongestListOfChars]

        if Main.showSteps == True: 
            imgContours = np.zeros((height, width, 3), np.uint8)
            del contours[:]

            for matchingChar in longestListOfMatchingCharsInPlate:
                contours.append(matchingChar.contour)
            # end for

            cv2.drawContours(imgContours, contours, -1, Main.SCALAR_WHITE)

            cv2.imshow("9", imgContours)
        # end if

        possiblePlate.strChars = recognizeCharsInPlate(possiblePlate.imgThresh, longestListOfMatchingCharsInPlate)

        if Main.showSteps == True:
            print("chars found in plate number " + str(
                intPlateCounter) + " = " + possiblePlate.strChars + ", click on any image and press a key to continue . . .")
            intPlateCounter = intPlateCounter + 1
            cv2.waitKey(0)
        # end if 

    if Main.showSteps == True:
        print("\nchar detection complete, click on any image and press a key to continue . . .\n")
        cv2.waitKey(0)
    # end if

    return listOfPossiblePlates
# end function

###################################################################################################
def findPossibleCharsInPlate(imgGrayscale, imgThresh):
    listOfPossibleChars = []                        # povratna vrijednost
    contours = []
    imgThreshCopy = imgThresh.copy()

            # pronalazak svih kontura na slici
    contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:                        # za svaku konturu
        possibleChar = PossibleChar.PossibleChar(contour)

        if checkIfPossibleChar(possibleChar):
            listOfPossibleChars.append(possibleChar)
        # end if
    # end if

    return listOfPossibleChars
# end function

###################################################################################################
def checkIfPossibleChar(possibleChar):
            # provjerava za svaku konturu da li je moguci char
            # jos uvijek se ne porede dva chara za slicnost
    if (possibleChar.intBoundingRectArea > MIN_PIXEL_AREA and
        possibleChar.intBoundingRectWidth > MIN_PIXEL_WIDTH and possibleChar.intBoundingRectHeight > MIN_PIXEL_HEIGHT and
        MIN_ASPECT_RATIO < possibleChar.fltAspectRatio and possibleChar.fltAspectRatio < MAX_ASPECT_RATIO):
        return True
    else:
        return False
    # end if
# end function

###################################################################################################
def findListOfListsOfMatchingChars(listOfPossibleChars):
            # cilj funkcije je da se jedna velika lista charova podijeli u listu lista
    listOfListsOfMatchingChars = []

    for possibleChar in listOfPossibleChars:                      
        listOfMatchingChars = findListOfMatchingChars(possibleChar, listOfPossibleChars)        # pronaci svaki char, u velikoj listi, koji je isti charu koji se posmatra

        listOfMatchingChars.append(possibleChar)                # dodati trenutni char u listu mogucih charova

        if len(listOfMatchingChars) < MIN_NUMBER_OF_MATCHING_CHARS:    
            continue                                                                          
        # end if                                              
        listOfListsOfMatchingChars.append(listOfMatchingChars)  
        listOfPossibleCharsWithCurrentMatchesRemoved = []

        # ukloniti trenutnu listu mogucih charova iz velike liste, da ih ne koristimo dva puta
        # koristi se nova velika lista, ne dira se stara
        listOfPossibleCharsWithCurrentMatchesRemoved = list(set(listOfPossibleChars) - set(listOfMatchingChars))

        recursiveListOfListsOfMatchingChars = findListOfListsOfMatchingChars(listOfPossibleCharsWithCurrentMatchesRemoved)      # rekurzivni poziv

        for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars:        # za svaku listu podudarajucih charova, dobivenih iz rekurzivnog poziva
            listOfListsOfMatchingChars.append(recursiveListOfMatchingChars)             # dodati na originalnu listu, nove liste podudarajucih charova
        # end for

        break       # exit for

    # end for

    return listOfListsOfMatchingChars
# end function

###################################################################################################
def findListOfMatchingChars(possibleChar, listOfChars):
            # cilj ove funkcije je ukoliko imamo moguci char i listu svih mogucih charova,
            # pronaci sve charove u listi koji su isti posmatranom, i vratiti te charove kao listu
    listOfMatchingChars = []                # povratna vrijednost

    for possibleMatchingChar in listOfChars:
        if possibleMatchingChar == possibleChar:                                                    
            continue                               
        # end if
        fltDistanceBetweenChars = distanceBetweenChars(possibleChar, possibleMatchingChar)
        fltAngleBetweenChars = angleBetweenChars(possibleChar, possibleMatchingChar)
        fltChangeInArea = float(abs(possibleMatchingChar.intBoundingRectArea - possibleChar.intBoundingRectArea)) / float(possibleChar.intBoundingRectArea)
        fltChangeInWidth = float(abs(possibleMatchingChar.intBoundingRectWidth - possibleChar.intBoundingRectWidth)) / float(possibleChar.intBoundingRectWidth)
        fltChangeInHeight = float(abs(possibleMatchingChar.intBoundingRectHeight - possibleChar.intBoundingRectHeight)) / float(possibleChar.intBoundingRectHeight)

                # check if chars match
        if (fltDistanceBetweenChars < (possibleChar.fltDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) and
            fltAngleBetweenChars < MAX_ANGLE_BETWEEN_CHARS and
            fltChangeInArea < MAX_CHANGE_IN_AREA and
            fltChangeInWidth < MAX_CHANGE_IN_WIDTH and
            fltChangeInHeight < MAX_CHANGE_IN_HEIGHT):

            listOfMatchingChars.append(possibleMatchingChar)        # ukoliko se dva chara podudaraju, dodati trenutni char u listu svih charova
        # end if
    # end for

    return listOfMatchingChars                 
# end function

###################################################################################################
# Pitagorina teorema da se izracuna udaljenost izmedu dva chara
def distanceBetweenChars(firstChar, secondChar):
    intX = abs(firstChar.intCenterX - secondChar.intCenterX)
    intY = abs(firstChar.intCenterY - secondChar.intCenterY)

    return math.sqrt((intX ** 2) + (intY ** 2))
# end function

###################################################################################################
# obicna trigonometrija (SOH CAH TOA) da se izracuna ugao izmedu charova
def angleBetweenChars(firstChar, secondChar):
    fltAdj = float(abs(firstChar.intCenterX - secondChar.intCenterX))
    fltOpp = float(abs(firstChar.intCenterY - secondChar.intCenterY))

    if fltAdj != 0.0:                           # provjera da se ne dijeli sa 0
        fltAngleInRad = math.atan(fltOpp / fltAdj)    
    else:
        fltAngleInRad = 1.5708             
    # end if

    fltAngleInDeg = fltAngleInRad * (180.0 / math.pi)       # izracunati ugao u stepenima

    return fltAngleInDeg
# end function

###################################################################################################
# ukoliko se dva chara preklapaju, treba ukloniti onaj manji
def removeInnerOverlappingChars(listOfMatchingChars):
    listOfMatchingCharsWithInnerCharRemoved = list(listOfMatchingChars)                

    for currentChar in listOfMatchingChars:
        for otherChar in listOfMatchingChars:
            if currentChar != otherChar:        # ukoliko trenutni char i drugi nisu isti                                       
                if distanceBetweenChars(currentChar, otherChar) < (currentChar.fltDiagonalSize * MIN_DIAG_SIZE_MULTIPLE_AWAY):
                                # pronadeni su preklapajuci charovi
                                # pronademo koji je char manji, ukoliko nije uklonjen ranije, uklonimo ga sad
                    if currentChar.intBoundingRectArea < otherChar.intBoundingRectArea:         # ukoliko je trenutni char manji od drugih
                        if currentChar in listOfMatchingCharsWithInnerCharRemoved:              # ukoliko trenutni char nije uklonjen u prethodnom prolazu
                            listOfMatchingCharsWithInnerCharRemoved.remove(currentChar)         # ukloni sad
                        # end if
                    else:                                                                       # ukoliko je trenutni char manji od drugih
                        if otherChar in listOfMatchingCharsWithInnerCharRemoved:                # ukoliko drugi char nije uklonjen u prethodnom prolazu
                            listOfMatchingCharsWithInnerCharRemoved.remove(otherChar)           # ukloni drugi char
                        # end if
                    # end if
                # end if
            # end if
        # end for
    # end for

    return listOfMatchingCharsWithInnerCharRemoved
# end function

###################################################################################################
# prepoznavanje charova
def recognizeCharsInPlate(imgThresh, listOfMatchingChars):
    strChars = ""

    height, width = imgThresh.shape

    imgThreshColor = np.zeros((height, width, 3), np.uint8)

    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)        # sortirati charove s lijeva na desno

    cv2.cvtColor(imgThresh, cv2.COLOR_GRAY2BGR, imgThreshColor)                     # napraviti threshold sliku u boji tako da mozemo nacrtati konturu

    for currentChar in listOfMatchingChars:                                   
        pt1 = (currentChar.intBoundingRectX, currentChar.intBoundingRectY)
        pt2 = ((currentChar.intBoundingRectX + currentChar.intBoundingRectWidth), (currentChar.intBoundingRectY + currentChar.intBoundingRectHeight))

        cv2.rectangle(imgThreshColor, pt1, pt2, Main.SCALAR_GREEN, 2)           # zeleni kvadrat oko chara

                # izrezati char iz slike
        imgROI = imgThresh[currentChar.intBoundingRectY : currentChar.intBoundingRectY + currentChar.intBoundingRectHeight,
                           currentChar.intBoundingRectX : currentChar.intBoundingRectX + currentChar.intBoundingRectWidth]

        imgROIResized = cv2.resize(imgROI, (RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT))     

        npaROIResized = imgROIResized.reshape((1, RESIZED_CHAR_IMAGE_WIDTH * RESIZED_CHAR_IMAGE_HEIGHT))    

        npaROIResized = np.float32(npaROIResized)              

        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 1)            

        strCurrentChar = str(chr(int(npaResults[0][0])))            

        strChars = strChars + strCurrentChar                        # dodati trenutni char na string

    # end for

    if Main.showSteps == True: 
        cv2.imshow("10", imgThreshColor)
    return strChars
# end function








