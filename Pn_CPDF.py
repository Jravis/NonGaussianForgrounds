import numpy as np
from multiprocessing import Process


def Hist(data, mbin, loop):
    cfrq = []
    frq = []
    j = 0
    Count = len(data)
    for j in xrange(len(mbin)):
        count1 = 0
        count2 = 0
        for ii in xrange(len(data)):
            if mbin[j] < data[ii]:
                count1 += 1
        for ii in xrange(len(data)):
            if j !=(len(mbin)-1):
                if mbin[j] <= data[ii] < mbin[j+1]:
                    count2 += 1
        cfrq.append(count1)
        frq.append(count2)

    frq = np.asarray(frq, dtype=float)
    frq = (frq*1.0)/Count

    name = '/home/sandeep/Parllel_Heslam/128/CPDF/Frq%d.txt'% (loop)
    with open(name,'w')  as f:
        for i in xrange(len(frq)):
            f.write("%f\n" % frq[i])
    name = '/home/sandeep/Parllel_Heslam/128/CPDF/bins%d.txt'% (loop)
    with open(name,'w')  as f:
        for i in xrange(len(mbin)):
            f.write("%f\n" % mbin[i])


if __name__ == "__main__":
    
    Pn1 = np.genfromtxt("/home/sandeep/Parllel_Heslam/128/Haslam128_All_output512_Degreereso.txt")
    """
    Pn2 = np.genfromtxt("/home/sandeep/Parllel_Heslam/Haslam_256/Haslam256_All_output2_Degreereso.txt")
    Pn3 = np.genfromtxt("/home/sandeep/Parllel_Heslam/Haslam_256/Haslam256_All_output3_Degreereso.txt")
    Pn4 = np.genfromtxt("/home/sandeep/Parllel_Heslam/Haslam_256/Haslam256_All_output4_Degreereso.txt")
    Pn5 = np.genfromtxt("/home/sandeep/Parllel_Heslam/Haslam_256/Haslam256_All_output6_Degreereso.txt")
    Pn6 = np.genfromtxt("/home/sandeep/Parllel_Heslam/Haslam_256/Haslam256_All_output8_Degreereso.txt")
    Pn7 = np.genfromtxt("/home/sandeep/Parllel_Heslam/Haslam_256/Haslam256_All_output10_Degreereso.txt")
    """
    bin_width1 = (max(Pn1) - min(Pn1))/100
    """
    bin_width2 = (max(Pn2) - min(Pn2))/100
    bin_width3 = (max(Pn3) - min(Pn3))/100
    bin_width4 = (max(Pn4) - min(Pn4))/100
    bin_width5 = (max(Pn5) - min(Pn5))/100
    bin_width6 = (max(Pn6) - min(Pn6))/100
    bin_width7 = (max(Pn7) - min(Pn7))/100
    """
    print max(Pn1) , min(Pn1)
    """
    print max(Pn2) , min(Pn2)
    print max(Pn3) , min(Pn3)
    print max(Pn4) , min(Pn4)
    """
    
    bins1 = np.arange(np.amin(Pn1),  np.amax(Pn1)+bin_width1, bin_width1)
    """
    bins2 = np.arange(np.amin(Pn2),  np.amax(Pn2)+bin_width2, bin_width2)
    bins3 = np.arange(np.amin(Pn3),  np.amax(Pn3)+bin_width3, bin_width3)
    bins4 = np.arange(np.amin(Pn4),  np.amax(Pn4)+bin_width4, bin_width4)
    bins5 = np.arange(np.amin(Pn5),  np.amax(Pn5)+bin_width5, bin_width5)
    bins6 = np.arange(np.amin(Pn6),  np.amax(Pn6)+bin_width6, bin_width6)
    bins7 = np.arange(np.amin(Pn7),  np.amax(Pn7)+bin_width7, bin_width7)
    print len(bins1), len(bins3), len(bins2)
    """

    Cell_Count1 = Process(target=Hist, args=(Pn1, bins1, 1))
    Cell_Count1.start()

    Cell_Count1.join()
    """
    Cell_Count2 = Process(target=Hist, args=(Pn2, bins2, 2))
    Cell_Count2.start()
    Cell_Count3 = Process(target=Hist, args=(Pn3, bins3, 3))
    Cell_Count3.start()
    Cell_Count4 = Process(target=Hist, args=(Pn4, bins4, 4))
    Cell_Count4.start()
    Cell_Count5 = Process(target=Hist, args=(Pn5, bins5, 6))
    Cell_Count5.start()
    Cell_Count6 = Process(target=Hist, args=(Pn6, bins6, 8))
    Cell_Count6.start()
    Cell_Count7 = Process(target=Hist, args=(Pn7, bins7, 10))
    Cell_Count7.start()

    Cell_Count2.join()
    Cell_Count3.join()
    Cell_Count4.join()
    Cell_Count5.join()
    Cell_Count6.join()
    Cell_Count7.join()
    """


