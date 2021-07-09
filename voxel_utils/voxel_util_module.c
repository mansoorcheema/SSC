#define PY_SSIZE_T_CLEAN
#include <stdio.h>
#include <Python.h>
#include <numpy/arrayobject.h>

extern double * myVectorAdd(double * h_A, double * h_B, int numElements);
extern void ComputeTSDF(double * cam_info_CPU, double * vox_info_CPU, double * depth_data_CPU,  double * vox_tsdf_CPU, double * depth_mapping_idxs_CPU, double * occupancy);


int sum(int a, int b ) { 
    return a+b;
}


static PyObject* compute_tsdf(PyObject* self, PyObject *args) {

    PyArrayObject* cam_info_CPU, * vox_info_CPU, * depth_data_CPU, * vox_tsdf_CPU, * depth_mapping_idxs_CPU, *occupancy;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!", &PyArray_Type, &cam_info_CPU,  &PyArray_Type, &vox_info_CPU,  &PyArray_Type, &depth_data_CPU,   &PyArray_Type, &vox_tsdf_CPU, &PyArray_Type, &depth_mapping_idxs_CPU,  &PyArray_Type, &occupancy )) {
    	PyErr_SetString(PyExc_ValueError, "Failed to parse arguments");
    	return NULL;
    	
    	}
    
     printf("Arguments parsed\n---------------\n");
     
     printf ("cam_info: %d\n", cam_info_CPU->dimensions[0]);
     printf ("vox_info: %d\n", vox_info_CPU->dimensions[0]);
     printf ("depth_data: %d\n", depth_data_CPU->dimensions[0]);
     printf ("vox_tsdf: %d\n", vox_tsdf_CPU->dimensions[0]);
     printf ("depth_mapping_idxs: %d\n", depth_mapping_idxs_CPU->dimensions[0]);
     printf ("occupancy: %d\n", occupancy->dimensions[0]);
     
     if (cam_info_CPU -> nd != 1 || vox_info_CPU -> nd != 1 || cam_info_CPU->descr->type_num != PyArray_DOUBLE || vox_info_CPU->descr->type_num != PyArray_DOUBLE || depth_data_CPU->descr->type_num != PyArray_DOUBLE || vox_tsdf_CPU->descr->type_num != PyArray_DOUBLE || depth_mapping_idxs_CPU->descr->type_num != PyArray_DOUBLE || occupancy->descr->type_num != PyArray_DOUBLE) {
        PyErr_SetString(PyExc_ValueError, "arrays must be one-dimensional and of type float");
        return NULL;
    }
     
     ComputeTSDF((double *) cam_info_CPU -> data,(double *) vox_info_CPU -> data,(double *) depth_data_CPU -> data,(double *) vox_tsdf_CPU -> data,(double *) depth_mapping_idxs_CPU -> data, (double *) occupancy -> data);
    
    return Py_None;
    
}


static PyObject* calculate_sum(PyObject* self, PyObject *args) {

    int a,b;
    PyArg_ParseTuple(args, "ii", &a, &b);
    
    //PyArg_ParseTuple(args, "i", &b);
    
    
    PyObject* value = PyLong_FromLong(sum(a,b));
    return Py_BuildValue("N", value);
    
}

static PyObject* vector_add(PyObject* self, PyObject* args) {
    PyArrayObject* array1, * array2;
    printf("parsing elements\n");
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &array1, &PyArray_Type, &array2))
        return NULL;

    
     printf("elements parsed\n");
     
    if (array1 -> nd != 1 || array2 -> nd != 1 || array1->descr->type_num != PyArray_DOUBLE || array2->descr->type_num != PyArray_DOUBLE) {
        PyErr_SetString(PyExc_ValueError, "arrays must be one-dimensional and of type float");
        return NULL;
    }
    

    int n1 = array1->dimensions[0];
    int n2 = array2->dimensions[0];

    printf("running vector_add on dim1: %d, stride1: %d, dim2: %d, stride2: %d\n", n1, array1->strides[0], n2, array2->strides[0]);

    if (n1 != n2) {
        PyErr_SetString(PyExc_ValueError, "arrays must have the same length");
        return NULL;
    }
    
    double * output = myVectorAdd((double *) array1 -> data, (double *) array2 -> data, n1);
    /*
    double * output = (double *) malloc(sizeof(double) * n1);

    for (int i = 0; i < n1; i++)
        output[i] = *((double *) array1 -> data + i) + *((double *) array2 -> data + i);
        */

    return PyArray_SimpleNewFromData(1, PyArray_DIMS(array1), PyArray_TYPE(array1), output);
}

static PyObject* helloworld(PyObject* self, PyObject* args) {
    printf("Hello World\n");
    return Py_None;
}

static PyMethodDef methods[] = {
    {"helloworld", helloworld, METH_NOARGS, "A Simple Hello World Function"}, // (function name, function, arguments, doc_string)
    {"sum", calculate_sum, METH_VARARGS, "Calculate sum" },
    //{"vector_add", vector_add, METH_VARARGS, "Calculate numpy array sum" },
    {"compute_tsdf", compute_tsdf, METH_VARARGS, "Calculate TSDF from Depth Map" },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef voxel_util_module = {
    PyModuleDef_HEAD_INIT, "VoxelUtils", // name of the module
    "VoxelUtils", -1, methods
};

PyMODINIT_FUNC PyInit_VoxelUtils(void) {
    import_array()
    return PyModule_Create(&voxel_util_module);
}
