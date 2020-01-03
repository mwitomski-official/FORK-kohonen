import cv2
import numpy as np
import random as rand

path = 'obrazy/lena.png'
image = cv2.imread(path)

height, width, channels = image.shape
print(height, width, channels)

dim = (512, 512)
resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
cv2.imshow("", resized)
(Blue, Green, Red) = cv2.split(resized)
# cv2.imshow("Red", R)
# cv2.imshow("Green", G)
# cv2.imshow("Blue", B)

frame = 2 # rozmiar pojedynczej ramki

ile_blokow = int(((512/frame)**2))
print(ile_blokow)

puzzle = [] # rameczki

for x in range(0, 511, frame):
	for y in range(0, 511, frame):
		puzzle.append(Red[x:x+frame,y:y+frame])	# ? .flatten()

puzzle = np.array(puzzle)
#print(puzzle[0])

# tutaj budujemy macierz wektorów normalizowanych, aby nie liczyć wielokrotnie

def normalizacja_wektora(input):
	if sum(input.flatten()) != 0:
		return input.flatten()/sum(input.flatten())
	else:
		return 0

puzzle_norm = []
for i in range(ile_blokow):
	puzzle_norm.append(normalizacja_wektora(puzzle[i]))

# otrzymujemy tablicę spłaszczonych wektorów normalizowanych puzzle_norm
# budujemy macierz neuronów "startowych", czyli tak naprawdę losujemy indexy ramek z puzzle[]

ile_neuronow = 8
neurony = []

for i in range(ile_neuronow):
	r = rand.randint(0,ile_blokow)
	neurony.append(puzzle_norm[r])

print("Wyznaczono neurony")

ile_ramek_treningowych = 8
training_set = []

for i in range(ile_ramek_treningowych):
	r = rand.randint(0,ile_ramek_treningowych)
	training_set.append(puzzle_norm[r])

print("Wyznaczono training set")

# dla każdego neuronu znajdujemy najbliższą ramkę z training setu, wyznaczamy Best Matching Unit 

for i in range(ile_ramek_treningowych):
	for j in range(ile_neuronow):
		print(sum(training_set[i]*neurony[j]))



print(training_set[0])

print(sum(neurony[0]*training_set[0]))
print(sum(neurony[1]*training_set[0]))
print(sum(neurony[2]*training_set[0]))
print(sum(neurony[3]*training_set[0]))
print(sum(neurony[4]*training_set[0]))
print(sum(neurony[5]*training_set[0]))
print(sum(neurony[6]*training_set[0]))
print(sum(neurony[7]*training_set[0]))



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
