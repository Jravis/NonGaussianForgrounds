import matplotlib.pyplot as plt
import numpy as np


filename = '/dataspace/sandeep/Bispectrum_data/fnl_test/100_fnl_100_Bispectrum/fnl_1_Bispectrum_%d.txt' % 900
Bis_obs = np.genfromtxt(filename, usecols=0, delimiter=',')
Bis_obs *= 2.7522**2


print len(Bis_obs)
esti_bis = np.zeros((899, 644), dtype=np.float64)

count = 0
for fn in xrange(1, 900):
    filename = '/dataspace/sandeep/Bispectrum_data/fnl_test/900_fnl_1_Bispectrum/fnl_1_Bispectrum_%d.txt' % fn
    tem = np.genfromtxt(filename, usecols=0, delimiter=',')
    esti_bis[count, :] = tem*2.7522**2
    count += 1


Bis_templet = np.mean(esti_bis, 0, dtype=np.float64)
Bis_var = np.square(np.std(esti_bis, 0, dtype=np.float64))

dino = np.sum(np.divide(np.square(Bis_templet), Bis_var))

print dino
fnl = np.sum(np.divide(np.divide(np.multiply(Bis_obs, Bis_templet), Bis_var), dino))

print fnl


sum = 0
for i in xrange(len(Bis_obs)):
    sum += (Bis_templet[i]*Bis_obs[i])/Bis_var[i]
print sum
print sum/dino



