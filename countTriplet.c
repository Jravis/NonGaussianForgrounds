#include"countTriplet.h"
#include<math.h>


int countTriplet(int *bin_1, int *bin_2, int *bin_3)
{
    int count = 0,l3=0, l2=0, l1=0;

    for(l3= bin_1[0]; l3 <  bin_1[1]+1; l3++)
    {
        for(l2= bin_2[0]; l2 <  bin_2[1]+1; l2++)
        {
            for (l1= bin_3[0]; l1 <  bin_3[1]+1; l1++)
            {
                if (fabs(l2-l1) <= l3 && l3<= l2+l1)
                {
                    if ((l3+l2+l1) % 2 == 0)
                    {
                        count++;
                    }
                }
            }
        }
    }
    return count;
}






