import cv2
import numpy as np
import random as rand

#path = 'obrazy/lena.png'
#path = 'obrazy/parrot.png'
path = 'obrazy/rm.png'

frame = 4 # rozmiar pojedynczej ramki
ile_neuronow = 48
ile_ramek_treningowych = 2048
eta = 0.1

epoki = 100

image = cv2.imread(path)
height, width, channels = image.shape

# print("Ścieżka obrazu: %s"% path)
# print("wysokość: %s, szerokość: %s, kanały: %s"% height, width, channels)
# print("\n")
# print("Rozmiar ramki: %s, Neurony: %s, Training set: %s, Eta: %s"% frame, ile_neuronow, ile_ramek_treningowych, eta)

def v_norm(input):
    norm = np.linalg.norm(input)
    if norm == 0: 
       return input
    return input / norm

ile_wierszy = int(512/frame)
ile_kolumn = int(512/frame)

#ile_blokow = int(((512/frame)**2))
ile_blokow = ile_wierszy*ile_kolumn

dim = (512, 512)
resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
#cv2.imshow("", resized)
(Blue, Green, Red) = cv2.split(resized)

print("Obraz podzielono na %s bloków"% ile_blokow)

# Podział tablicy wg parametru frame na mniejsze kwadraty
puzzle = [] 
for x in range(0, 511, frame):
	for y in range(0, 511, frame):
		puzzle.append(Red[x:x+frame,y:y+frame])	

# pobierane wartości są typu integer, jednak w toku obliczeń pojawia się typ float - aby nie utracić informacji, zmieniamy typ danych w tablicy puzzle na float
puzzle = np.array(puzzle).astype(float)

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

# dla każdego neuronu znajdujemy najbliższą ramkę z training setu, wyznaczamy Best Matching Unit 
for k in range(epoki):
	BMU = []
	for i in range(ile_neuronow):
		distance = []
		for j in range(ile_ramek_treningowych):
			dist = np.linalg.norm(neurony[i] - training_set[j])
			odl = [i, j, dist]
			distance.append(odl)
		BMU.append((min(distance, key=lambda x: x[2])))	
	#print(BMU)		

	for each in range(len(BMU)):
		new_neuron = neurony[BMU[each][0]] + eta * (training_set[BMU[each][1]] - neurony[BMU[each][0]])
		neurony[BMU[each][0]] = new_neuron
		
#print("Nauczone neurony")
# print(neurony)

# Nauczone neurony w intach i bez dubli
slownik = [list(x) for x in set(tuple(x) for x in neurony.astype(int))]
#print(*slownik)

slownik = np.array(slownik)
print("Wygenerowano słownik złożony z %s elementów."% len(slownik))

indeksy = []
for i in range(len(puzzle)):
	norm_ramka = v_norm(puzzle[i].flatten())
	winner = []
	for j in range(len(slownik)):
		norm_neuron = v_norm(slownik[j])
		winner.append([i, j, norm_neuron.dot(norm_ramka)])
	indeksy.append((max(winner, key=lambda x: x[2]))[1]) # tabela indeksów ze wskazaniem na element [1] ze zwycieskiej puli


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

#print(wiersze[2].shape)


final_picture = np.concatenate(wiersze, axis=0)

cv2.imshow("Final_picture", final_picture)

cv2.waitKey(0)
cv2.destroyAllWindows()
