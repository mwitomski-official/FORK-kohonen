import cv2, math, time
import random as rand

import numpy as np
from numpy import savetxt

import numba
from numba import jit

#******************************************************************************
#*********************    PARAMETRY URUCHOMIENIOWE   **************************
#******************************************************************************
seed = 500
np.random.seed(seed)

zapis_do_pliku = True

frame = 4 # rozmiar pojedynczej ramki x na x, min. 2
ile_neuronow = 8 # mi 1 max 255
ile_ramek_treningowych = 512
eta = 0.1
epoki = 150
path = 'obrazy/lena.png'

float_fake_bitsize = 4 # liczba bitów niezbędna do kodowania liczb typu float w rozwiązaniu praktycznym

# wymuszona rozdzielczość obrazu źródłowego, można pobrać z danych obrazowych
width = 512
height = 512

# używamy logarytmu o podstawie 2 oraz funkcji ceiling (zwraca najbliższą liczbę całkowitą - większą lub równą wskazanej)
# do obliczenia minimalnej liczby bitów niezbędnych do zakodowania indeksów
b = math.ceil(math.log(ile_neuronow,2)) 
 
ile_wierszy = int(height/frame)
ile_kolumn = int(width/frame)
ile_blokow = ile_wierszy*ile_kolumn

czas = round(time.time(),0)

try:
    file_log = open ("Kohonen_LOG.txt", "a+")
except IOError:
    print("Błąd odczytu pliku logowania")

#******************************************************************************
#*********************    FUNKCJE   *******************************************
#******************************************************************************

# dlugosc wektora = pierwiastek z sumy kwadratów (zwraca float)
@jit(nopython=True)
def v_sum(input):
	#norm = np.linalg.norm(input)
	output = 0
	for i in range(len(input)):
		output = output + input[i]**2
	output = output**0.5
	return output

# wektor znormalizowany (zwraca tablicę)
@jit(nopython=True)
def v_norm(input):
	if v_sum(input): 
		return input / v_sum(input)
	else:
		return input / 1

# euklidesowa odleglosc wektorow (zwraca wartosc float)
@jit(nopython=True)
def v_dist(input_1, input_2):
	return (v_sum(input_1-input_2))

# podobienstwo wektorów - zwraca stopień podobieństwa dwóch wektorów (od 0 (brak) do 1 (pełne podobieństwo))
@jit(nopython=True)
def v_simil(input_1, input_2):
	simil = 0
	iloczyn = input_1*input_2
	for i in range(len(iloczyn)):
		simil = simil+iloczyn[i]
	return simil

#******************************************************************************
#*********************    PRZEKSZTAŁCENIA OBRAZU   ****************************
#******************************************************************************

image = cv2.imread(path)
resized = cv2.resize(image, (height, width), interpolation = cv2.INTER_AREA)
(Blue, Green, Red) = cv2.split(resized)

puzzle = [] 
for x in range(0, width-1, frame):
	for y in range(0, height-1, frame):
		puzzle.append((Red[x:x+frame,y:y+frame]).flatten())

puzzle = np.array(puzzle)

puzzle_bright = []
puzzle_norm = []
for i in range(len(puzzle)):
	puzzle_bright.append(v_sum(puzzle[i]))
	puzzle_norm.append(v_norm(puzzle[i]))

puzzle_bright = np.array(puzzle_bright)
puzzle_norm = np.array(puzzle_norm)


#******************************************************************************
#********************    LOSOWANIE RAMEK TRENINGOWYCH   ***********************
#******************************************************************************

training_set = []
for i in range(ile_ramek_treningowych):
	r = np.random.randint(low=0, high = ile_ramek_treningowych)
	training_set.append(r)

training_set = np.array(training_set)

#******************************************************************************
#*********************    LOSOWANIE WAG NEURONÓW      *************************
#******************************************************************************

neurony = []
for i in range(ile_neuronow):
	neurony.append(np.random.uniform(0,254,[frame**2]))

neurony = np.array(neurony)

#******************************************************************************
#*********************    RAPORT Z PIERWSZEGO ETAPU  **************************
#******************************************************************************

print("Rozmiar ramki: %sx%s px; Neurony %s; Training set: %s; Eta: %s; Epok: %s; Seed: %s;"% (frame, frame, ile_neuronow, ile_ramek_treningowych, eta, epoki, seed))
print("Obraz podzielono na %s bloków"% ile_blokow)

print("Wyznaczono training set: %s"% ile_ramek_treningowych)
print("Wyznaczono neurony: %s"% ile_neuronow)

#******************************************************************************
#*********************    PĘTLA UCZĄCA  ***************************************
#******************************************************************************

for k in range(epoki):
	for i in range(len(training_set)): # dla każdej ramki
		temp_bmu =[]
		for j in range(len(neurony)): # wyszukujemy najbliższy neuron
			temp_dist = v_dist(puzzle_norm[training_set[i]], v_norm(neurony[j]))
		temp_bmu.append([i, j, temp_dist]) # utwórz tablicę [indeks ramki, indeks neuronu, odległość]
	min_dist = min(temp_bmu, key=lambda x: x[2]) # wybierz rekord o minimalnej długości
	if (k%10==0): print(epoki-k, min_dist)

	new_neuron = v_norm(neurony[min_dist[1]]) + eta * (puzzle_norm[min_dist[0]] - v_norm(neurony[min_dist[1]]))
	neurony[min_dist[1]] = new_neuron

print("\nKoniec nauki, wygenerowano tablicę neuronów.")

slownik=[]
for i in range(len(neurony)):
	slownik.append(v_norm(neurony[i]))

print("Wygenerowano słownik złożony z %s elementów."% len(slownik))
#print(*slownik[0])
print("Uczenie neuronów i wygenerowanie słownika zakończone.")

#******************************************************************************
#*********************    INDEKSOWANIE (KOMPRESJA)   **************************
#******************************************************************************

indeksy = []
for i in range(len(puzzle)):
	winner = []
	for j in range(len(slownik)):
		winner.append([i, j, v_simil(puzzle_norm[i], slownik[j])])
	indeksy.append((max(winner, key=lambda x: x[2]))[1]) # tabela indeksów ze wskazaniem na element [1] ze zwycieskiej puli

print("Zakończono indeksowanie wszystkich ramek obrazu.")

#******************************************************************************
#*********************    DEKOMPRESJA  ****************************************
#******************************************************************************

wiersz_long=[]
for i in range(len(indeksy)):
	#dekompres = slownik[indeksy[i]]*(v_sum(puzzle[i].flatten()))
	dekompres = slownik[indeksy[i]]*(puzzle_bright[i])
	wiersz_long.append(np.uint8(dekompres.reshape(frame, frame)))

wiersz = np.concatenate(wiersz_long, axis=1)

wiersze = [] 
for x in range(0, wiersz.shape[1], frame*ile_kolumn):
	wiersze.append(wiersz[:,x:x+(frame*ile_kolumn)])	

final_picture = np.concatenate(wiersze, axis=0)

#******************************************************************************
#*********************    RAPORT Z DRUGIEGO ETAPU     *************************
#******************************************************************************

bitsize_source = width*height*8 # liczba pikseli, piksel kodowany w 8 bitach

# założenia: w Python3 int to 4 bajty (32 bity), float to 8 bajtów (64 bity)
# przyjęto jasność ramek w liczbach całkowitych to 16 bitów, w liczbach float: 32 bity 
# neurony: jeden neuron: frame^2* 64 bit (float) razy liczba neuronów
# indeksy: liczby całkowite do 255, 8 bitów

bitsize_bright = len(puzzle_bright)*(float_fake_bitsize)
bitsize_neurony = len(neurony)*((frame**2)*float_fake_bitsize)
bitsize_indeksy = len(indeksy)*b

bitsize_comppressed = bitsize_bright+bitsize_neurony+bitsize_indeksy
WK = bitsize_source/bitsize_comppressed

print("Współczynnik kompresji %s"% WK)
print("całkowita liczba bitów dla obrazu oryginalnego (%s b) do liczby bitów dla obrazu skompresowanego (%s b)"% (bitsize_source, bitsize_comppressed))

MSE = ((1/width**2)*np.sum(((Red-final_picture)**2))) 
PSNR = 10 * np.log10(255**2/MSE)

print("Mean Square Error =  %s"% MSE)
print("Peak Signal-to-Noise Ratio =  %s dB"% PSNR)

# ********** Wyświetlenie obrazu i zapis ***************************************

print("Złożenie ramek w finalny obraz.")
cv2.imshow("Final_picture", final_picture)

stop = time.time()

filename = ("R %s; N %s; TS %s; Eta %s; Epok %s; Seed %s; WK %s; MSE %s; PSNR %s; timetag %s"% (frame, ile_neuronow, ile_ramek_treningowych, eta, epoki, seed, round(WK,2), round(MSE,2), round(PSNR,2), czas))

if zapis_do_pliku:
	file_log.writelines("%s; %s; %s; %s; %s; %s; %s; %s; %s; %s; %s\n"% (path, frame, ile_neuronow, ile_ramek_treningowych, eta, epoki, seed, WK, MSE, PSNR, czas))
	file_log.close()

#savetxt(filename+" indeksy.txt", np.uint8(indeksy), delimiter=';')
#savetxt(filename+" neurony.txt", np.uint8(neurony), delimiter=';')
#savetxt(filename+" final_picture.txt", np.uint8(final_picture), delimiter=';')
if zapis_do_pliku:
	cv2.imwrite(filename+".png", final_picture)


cv2.waitKey(0)
cv2.destroyAllWindows()
