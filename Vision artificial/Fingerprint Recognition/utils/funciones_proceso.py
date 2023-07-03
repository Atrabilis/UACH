import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import csv

# Limpiar la pantalla
os.system("cls")

# Funciones de procesamiento de imágenes
def fft_enhance(image, f):
    I = 255 - image.astype(np.float64)
    w, h = I.shape
    w1 = w // 32 * 32
    h1 = h // 32 * 32
    inner = np.zeros((w1, h1))
    
    for i in range(0, w1, 32):
        for j in range(0, h1, 32):
            a = i + 31
            b = j + 31
            F = np.fft.fft2(I[i:a, j:b])
            factor = np.abs(F) ** f
            block = np.abs(np.fft.ifft2(F * factor))
            larv = np.max(block)
            
            if larv == 0:
                larv = 1
            
            block = block / larv
            inner[i:a, j:b] = block
    
    final = inner * 255
    final = cv2.equalizeHist(final.astype(np.uint8))
    
    return final

def adaptiveBinarization(image, block_size):
    binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, 0)
    return binary_image