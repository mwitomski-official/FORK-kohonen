# Kohonen - testy metod CUDA na prostych wektorach

import numba
from numba import jit
import numpy as np


# dlugosc wektora = pierwiastek z sumy kwadratów (jedna wartość float)
@jit(nopython=True)
def v_sum(input):
	#norm = np.linalg.norm(input)
	output = 0
	for i in range(len(input)):
		output = output + input[i]**2
	output = output**0.5
	return output

# wektor znormalizowany (tablica)
@jit(nopython=True)
def v_norm(input):
	if v_sum(input): 
		return input / v_sum(input)
	else:
		return input / 1

@jit(nopython=True)
def v_dist(input_1, input_2):
	return (v_sum((input_1-input_2)))

@jit(nopython=True)
def v_simil(input_1, input_2):
	simil = 0
	iloczyn = v_norm(input_1)*v_norm(input_2)
	for i in range(len(iloczyn)):
		simil = simil+iloczyn[i]
	return simil

a = [2,3,4,5]
b = [2,3,4,5]
c = [1,1,8,8]

a = np.asarray(a)
b = np.asarray(b)
c = np.asarray(c)

print("Normalizacja cuda a:")
print(v_sum(a))
print("Normalizacja linalg.norm a:")
print(np.linalg.norm(a))

print("Normalizacja cuda b:")
print(v_sum(b))
print("Normalizacja linalg.norm b:")
print(np.linalg.norm(b))

print("Normalizacja cuda c:")
print(v_sum(c))
print("Normalizacja linalg.norm c:")
print(np.linalg.norm(c))

print("***************************************")

print("Odleglosci neuronów CUDA: AB i AC")
print(v_dist(a,b))
print(v_dist(a,c))

print("Odleglosci neuronów linalg: AB i AC")
print(np.linalg.norm(a-b))
print(np.linalg.norm(a-c))

print("***************************************")

print("Podobienstwo neuronow po normalizacji AB i AC")
print("A: %s"% v_norm(a))
print("B: %s"% v_norm(b))
print("C: %s"% v_norm(c))

print("Metoda .dot")
print(v_norm(a).dot(v_norm(b)))
print(v_norm(a).dot(v_norm(c)))

print("Metoda wlasna")
print("A podobne do B: %s"% v_simil(a,b))
print("A podobne do C: %s"% v_simil(a,c))
