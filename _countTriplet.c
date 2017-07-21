#include<Python.h>
#include<numpy/arrayobject.h>
#include "countTriplet.h"


static char module_docstring[] = "this module provides an interface for calculating chi-squared using c.";
static char countTriplet_docstring[] = "calculate the valid triplet in a l bins.";


static PyObject *countTriplet_countTriplet(PyObject *self, PyObject *args);


static PyMethodDef module_methods[] = 
{
    {"countTriplet", countTriplet_countTriplet, METH_VARARGS, countTriplet_docstring},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC init_countTriplet(void)
{
    PyObject *m = Py_InitModule3("_countTriplet", module_methods, module_docstring);
    if (m == NULL)
        return;

    import_array();
}


static PyObject *countTriplet_countTriplet(PyObject *self, PyObject *args)
{
    PyObject *bin1, *bin2, *bin3;

    if (!PyArg_ParseTuple(args, "OOO", &bin1, &bin2, &bin3))
        return NULL;

    PyObject *bin1_array = PyArray_FROM_OTF(bin1, NPY_INT, NPY_IN_ARRAY);
    PyObject *bin2_array = PyArray_FROM_OTF(bin2, NPY_INT, NPY_IN_ARRAY);
    PyObject *bin3_array = PyArray_FROM_OTF(bin3, NPY_INT, NPY_IN_ARRAY);

    if (bin1_array == NULL || bin2_array == NULL || bin3_array == NULL) 
    {
        Py_XDECREF(bin1_array);
        Py_XDECREF(bin2_array);
        Py_XDECREF(bin3_array);
        return NULL;
    }

    //int N = (int)PyArray_DIM(bin1_array, 0);

    int *arry1    = (int*)PyArray_DATA(bin1_array);
    int *arry2    = (int*)PyArray_DATA(bin2_array);
    int *arry3 = (int*)PyArray_DATA(bin3_array);

    int value = countTriplet(arry1, arry2, arry3);

    Py_DECREF(bin1_array);
    Py_DECREF(bin2_array);
    Py_DECREF(bin3_array);

    PyObject *ret = Py_BuildValue("i", value);
    return ret;
}






