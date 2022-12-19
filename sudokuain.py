import cv2
import numpy as np
from utlis import *


print('Settings UP')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
model = initializePredictionModel()


pathImage = 'Sudoku_test_3.png' # Путь до тестового изображения

# Шаг 1. Подготовка изображения
heightImg = 450
widthImg = 450
img = cv2.imread(pathImage) # Считываем изображение по нашему пути
img = cv2.resize(img, (widthImg, heightImg)) # Используем функцию изменения размера изображения, под необходимые нам
imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)
imgThreshold = preProcess(img) # Здесь мы используем самописную функцию из файла "utlis.py"

# Шаг 2. Поиск контуров
imgContours = img.copy() # Копируем изначальное изображение для преобразований
imgBigContours = img.copy() # Копируем изначальное изображение для преобразований
# Поиск контуров на изображении пропущенном через Treshold с помощью метода RETR_EXTERNAL
# затем мы используем CHAIN_APPROX_SIMPLE
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3) # Рисуем все контуры, которые смогли зафиксировать

# Шаг 3. Поиск самого большого контура и использование его в качестве поля для судоку
biggest, maxArea = biggestContour(contours) # Наша самописаная функция по поиску контура
if biggest.size != 0:
    biggest = reorder(biggest)
    cv2.drawContours(imgBigContours, biggest, -1, (0, 0, 255), 15) # Рисуем самый большой контур
    pts1 = np.float32(biggest)
    pts2 =np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]]) # Подготовка "точек"
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    imgDetectedDigits = imgBlank.copy()
    imgWarpColored = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)

#Шаг 4. Найдём на изображении каждую цифру
imgSolvedDigits = imgBlank.copy() # Копируем изначальное изображение для преобразований
boxes = splitBoxes(imgWarpColored) # Используем нашу самописную функцию и передаём туда вырезанное изображение поля в оттенках серово
#cv2.imshow("Sample", boxes[5])
numbers = getPrediction(boxes, model) # Используем самописную функцию предсказания цифр с нашей предобученной моделью
imgDetectedDigits = displayNumbers(imgDetectedDigits, numbers, color = (255, 0, 255)) # Опять же самописная функция для расположения цифр на картинке
numbers = np.asarray(numbers)
posArray = np.where(numbers > 0, 0, 1)


imageArray = ([img, imgThreshold, imgContours,  imgBigContours], [imgWarpColored, imgDetectedDigits, imgBlank, imgBlank]) # Массив изображений
#imageArray = ([img, img, img, img], [img, img, img, img]) # Массив изображений

stackedImage = stackImages(imageArray, 1)
cv2.imshow('Stacked Images', stackedImage) # Показ изображения

cv2.waitKey(0)