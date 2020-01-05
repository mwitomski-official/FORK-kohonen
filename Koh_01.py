import cv2
import numpy as np
import random as rand

path = 'obrazy/lena.png'
#path = 'obrazy/black.png'
#path = 'obrazy/black_3_white_squares.png'
#path = 'obrazy/white.png'

image = cv2.imread(path)

height, width, channels = image.shape
print(height, width, channels)


frame = 4 # rozmiar pojedynczej ramki
ile_neuronow = 4
ile_ramek_treningowych = 80

# współczynnik uczenia
eta = 0.01

ile_blokow = int(((512/frame)**2))

def normalizacja_wektora(input):
	if sum(input.flatten()) != 0:
		#return input.flatten()/sum(input.flatten())
		return input/sum(input)
	else:
		return 0

dim = (512, 512)
resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
cv2.imshow("", resized)
(Blue, Green, Red) = cv2.split(resized)

print("Obraz podzielono na %s bloków"% ile_blokow)

puzzle = [] # rameczki
for x in range(0, 511, frame):
	for y in range(0, 511, frame):
		puzzle.append(Red[x:x+frame,y:y+frame])	# ? .flatten()

puzzle = np.array(puzzle).astype(float)


# tutaj budujemy macierz wektorów normalizowanych, aby nie liczyć wielokrotnie - może się przyda?
# puzzle_norm = []
# for i in range(ile_blokow):
# 	puzzle_norm.append(normalizacja_wektora(puzzle[i]))

# otrzymujemy tablicę spłaszczonych wektorów normalizowanych puzzle_norm
# budujemy macierz neuronów "startowych", czyli tak naprawdę losujemy indexy ramek z puzzle[]

neurony = []
for i in range(ile_neuronow):
	r = rand.randint(0,	ile_blokow-1) 
	neurony.append(puzzle[r].flatten())

neurony = np.array(neurony)
print("Wyznaczono neurony: %s"% ile_neuronow)
print(neurony)
print(type(neurony))


training_set = []
for i in range(ile_ramek_treningowych):
	r = rand.randint(0,ile_ramek_treningowych)
	training_set.append(puzzle[r].flatten())

print("Wyznaczono training set: %s"% ile_ramek_treningowych)
training_set=np.array(training_set)
print(training_set)
print(type(training_set))

# dla każdego neuronu znajdujemy najbliższą ramkę z training setu, wyznaczamy Best Matching Unit 

for k in range(5):
	print("Krok %s"% k)
	BMU = []
	for i in range(ile_neuronow):
		distance = []
		for j in range(ile_ramek_treningowych):
			#print(i,j)
			# do porównania w locie normalizujemy wskazane wektory neuronu i ramki obrazu
			# odl = [i, j, sum(normalizacja_wektora(neurony[i])*(normalizacja_wektora((training_set[j]))))]
			# print("Neuron")
			# print(normalizacja_wektora(neurony[i]))
			# print("Ramka")
			# print(normalizacja_wektora((training_set[j])))
			suma_wazona_wek = normalizacja_wektora(neurony[i]).dot(normalizacja_wektora(training_set[j]))
			# print("Iloczyn")
			# print(iloczyn_wek)
			odl = [i, j, suma_wazona_wek]
			distance.append(odl)
		#distance = np.array(distance)
		BMU.append((min(distance, key=lambda x: x[2])))	
	print("BMU")
	print(type(BMU))
	
	for each in range(len(BMU)):
	 	#new_neuron = (neurony[BMU[each][0]] + eta * (neurony[BMU[each][0]]-puzzle[BMU[each][1]]))
		print("stary neuron")
		print(type(neurony[BMU[each][0]]))
		print("ramka z training set")
		print(training_set[BMU[each][1]])
		new_neuron = neurony[BMU[each][0]] + eta * (neurony[BMU[each][0]] - training_set[BMU[each][1]])
		print("nowy neuron")
		print(new_neuron)
		#new_neuron = np.array(new_neuron)
#		cos = [1.2,2.3,3,4,5,6,7,8,9,0,1,2,3,4,5,6]
#		new_neuron = new_neuron.tolist()
#		print(type(new_neuron))
		#print(cos)
		print("Przypisanie do tablicy neuronów")
		neurony[BMU[each][0]] = new_neuron
		print(neurony[BMU[each][0]])

	 	#print(neurony[BMU[each][0]])
	 	#print(puzzle[BMU[each][1]])
		
	 	
	 	
	 	#print(neurony[BMU[0][0]])


# array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15])
# >>> A = A.reshape(4,4)

# print("stary neuron")
# print(neurony[BMU[0][0]])

# print("ramka:")
# print(puzzle[BMU[0][1]])

# new_neuron = neurony[BMU[0][0]] + eta * (neurony[BMU[0][0]]-puzzle[BMU[0][1]])
# print("nowy neuron")
# print(new_neuron)


# print(training_set[0])

# print(sum(neurony[0]*training_set[0]))
# print(sum(neurony[1]*training_set[0]))
# print(sum(neurony[2]*training_set[0]))
# print(sum(neurony[3]*training_set[0]))
# print(sum(neurony[4]*training_set[0]))
# print(sum(neurony[5]*training_set[0]))
# print(sum(neurony[6]*training_set[0]))
# print(sum(neurony[7]*training_set[0]))



# mając neurony startowe, możemy zacząć proces "uczenia"



# print(puzzle_norm[14])
# print(puzzle_norm[24])
# print(puzzle_norm[48])

# cv2.imshow("14",puzzle[24])
# cv2.imshow("24",puzzle[22])
# cv2.imshow("48",puzzle[28])

#print(sum(puzzle[neuron].flatten()))

#x = rand.randrange(0, int((512/frame)**2), 1)
#print(x)

#cv2.imshow('Kafelek', puzzle[3])
# x.append(Red[0:4,0:4].flatten())
# x.append(Red[4:8,4:8].flatten())
# x.append(Red[8:12,8:12].flatten())
# x.append(Red[12:16,12:16].flatten())
# x.append(Red[16:20,16:20].flatten())

# print(x[0])
# print(x[1])
# print(x[2])
# print(x[3])
# print(x[4])

# print(sum(x[4].flatten()))
# y=[]


# z=[[1,1,1,1,1,1,1,1],
# 	[2,2,2,2,2,2,2,2],
# 	[3,3,3,3,3,3,3,3],
# 	[4,4,4,4,4,4,4,4]]

# z=np.array(z)
# xz=[]

# xz.append(z[0:4,0:4])


# #xz.append(z[4:8,4:8].flatten())

# print(xz)



cv2.waitKey(0)
cv2.destroyAllWindows()
