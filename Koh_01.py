import cv2
import numpy as np
import random as rand
import time

path = 'obrazy/lena.png'
#path = 'obrazy/boat.png'
#path = 'obrazy/rm.png'

frame = 2 # rozmiar pojedynczej ramki
ile_neuronow = 16
ile_ramek_treningowych = 16384
eta = 0.1

epoki = 1000

image = cv2.imread(path)
height, width, channels = image.shape

def v_norm(input):
    norm = np.linalg.norm(input)
    if norm == 0: 
       return input
    return input / norm

Full_start = time.time()

ile_wierszy = int(512/frame)
ile_kolumn = int(512/frame)

#ile_blokow = int(((512/frame)**2))
ile_blokow = ile_wierszy*ile_kolumn

dim = (512, 512)
resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
(Blue, Green, Red) = cv2.split(resized)

print("Ścieżka obrazu: %s"% path)
print("Wysokość: %s, szerokość: %s, kanały: %s"% (height, width, channels))
print("Wyodrębniono kanał Red jako reprezentację skali szarości, przeskalowano obraz do 512x512px.")
print("\n")
print("Rozmiar ramki: %sx%s px; Neurony %s; Training set: %s; Eta: %s; Epok: %s;"% (frame, frame, ile_neuronow, ile_ramek_treningowych, eta, epoki))

rozmiar = (ile_blokow*8)+(ile_neuronow*(frame**2)*8)
print("Przewidywany rozmiar indksowanego obrazu: %s bitów (%s bajtów)"% (rozmiar, rozmiar/8))

start = time.time()
# Podział tablicy wg parametru frame na mniejsze kwadraty
puzzle = [] 
for x in range(0, 511, frame):
	for y in range(0, 511, frame):
		puzzle.append(Red[x:x+frame,y:y+frame])	

# pobierane wartości są typu integer, jednak w toku obliczeń pojawia się typ float - aby nie utracić informacji, zmieniamy typ danych w tablicy puzzle na float
puzzle = np.array(puzzle).astype(float)
print("\n")
print("Obraz podzielono na %s bloków"% ile_blokow)

# losujemy wskazaną liczbę neuronów startowych
neurony = []
for i in range(ile_neuronow):
	r = rand.randint(0,	ile_blokow-1) 
	neurony.append(puzzle[r].flatten())

neurony = np.array(neurony)
print("Wyznaczono neurony: %s"% ile_neuronow)

# generujemy Training set - wskazaną liczbę ramek z obrazu źródłowego 
training_set = []
for i in range(ile_ramek_treningowych):
	r = rand.randint(0,ile_ramek_treningowych)
	training_set.append(puzzle[r].flatten())

print("Wyznaczono training set: %s"% ile_ramek_treningowych)
training_set=np.array(training_set)

end = time.time()
print("Generowanie neuronów startowych i ramek treningowych: %s sekund"% round(end-start,3))

start = time.time()
# dla każdego neuronu znajdujemy najbliższą ramkę z training setu, wyznaczamy Best Matching Unit i modyfikujemy neuron
for k in range(epoki):
	print(k)
	BMU = []
	for i in range(ile_neuronow):
		distance = []
		for j in range(ile_ramek_treningowych):
			dist = np.linalg.norm(neurony[i] - training_set[j])
			odl = [i, j, dist]
			distance.append(odl)
		BMU.append((min(distance, key=lambda x: x[2])))	

	for each in range(len(BMU)):
		new_neuron = neurony[BMU[each][0]] + eta * (training_set[BMU[each][1]] - neurony[BMU[each][0]])
		neurony[BMU[each][0]] = new_neuron


# Nauczone neurony w intach i bez dubli
slownik = [list(x) for x in set(tuple(x) for x in neurony.astype(int))]
#print(*slownik)

slownik = np.array(slownik)
print("Wygenerowano słownik złożony z %s elementów."% len(slownik))
end = time.time()
print("Uczenie neuronów i wygenerowanie słownika: %s sekund"% round(end-start,3))

start = time.time()
indeksy = []
for i in range(len(puzzle)):
	print(i)
	norm_ramka = v_norm(puzzle[i].flatten())
	winner = []
	for j in range(len(slownik)):
		norm_neuron = v_norm(slownik[j])
		winner.append([i, j, norm_neuron.dot(norm_ramka)])
	indeksy.append((max(winner, key=lambda x: x[2]))[1]) # tabela indeksów ze wskazaniem na element [1] ze zwycieskiej puli
end = time.time()
print("Indeksowanie wszystkich ramek obrazu: %s sekund"% round(end-start,3))

start = time.time()
# reshape 
wiersz_long=[]
for i in range(len(indeksy)):
	#print("i"+ str(i))
	wiersz_long.append(np.uint8(slownik[indeksy[i]]).reshape(frame, frame))

wiersz = np.concatenate(wiersz_long, axis=1)

#print(wiersz[:,:64].shape)

print(wiersz.shape)

wiersze = [] 
for x in range(0, wiersz.shape[1], frame*ile_kolumn):
	wiersze.append(wiersz[:,x:x+(frame*ile_kolumn)])	

final_picture = np.concatenate(wiersze, axis=0)
end = time.time()
print("Złożenie ramek w finalny obraz: %s sekund"% round(end-start,3))

Full_stop = time.time()
print("Całkowity czas trwania operacji: %s sekund"% round(Full_stop-Full_start,3))

cv2.imshow("Final_picture", final_picture)

cv2.waitKey(0)
cv2.destroyAllWindows()


