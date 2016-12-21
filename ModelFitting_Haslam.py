"""
In This routine is to fit gaussian to count probablity distribution function
"""
import numpy as np
import matplotlib.pyplot as plt


def gaussian(x, std, mean):
    """
    :param x:
    :param std:
    :param mean:
    :return:
    """
    return (1./(np.sqrt(2*np.pi)*std)) * np.exp(-0.5 * np.square((x-mean)/std))


def chisqfunc(data, bin, stdev, mean, w):
    """
    :param data:
    :param bin:
    :param stdev:
    :param mean:
    :param w:
    :return:
    """
    chisq = np.zeros(len(stdev), float)
    for i in xrange(len(stdev)):
        model = gaussian(bin, stdev[i], mean)
        chisq[i] = np.sum((np.multiply((data - model)**2, w)))
    return chisq

name ='/home/sandeep/Parllel_Heslam/Model_fitting_data/Frq_Bin_AllSky.txt'

Frq1 = np.genfromtxt(name, usecols=0)
bin1 = np.genfromtxt(name, usecols=1)
Weight = np.genfromtxt(name, usecols=2)

std1 = np.std(Frq1)
mean1 = np.mean(Frq1)

print std1, mean1
std_bins = np.linspace(std1-(std1*0.5), 100, 100)
print std_bins


y = chisqfunc(Frq1, bin1, std_bins, mean1, Weight)
print min(y)

plt.figure(1)
plt.plot(bin1, Frq1, "g*-")
plt.grid()
plt.yscale("log")
plt.xscale("log")
plt.figure(2)
plt.plot(std_bins, y, "g*-")
plt.grid()
#plt.yscale("log")
#plt.xscale("log")

plt.show()

