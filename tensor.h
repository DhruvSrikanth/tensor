#ifndef TENSOR_H
#define TENSOR_H

// Storage structure to manage the actual data and reference counting
typedef struct {
    float* data;
    int data_size;
    int ref_count;
} Storage;

// Tensor structure representing an n-dimensional tensor
typedef struct {
    Storage* storage;  // Pointer to the storage of the data
    int offset;        // Offset into the storage
    int ndim;          // Number of dimensions
    int* shape;        // Defines the size of each dimension
    int* strides;      // Defines how many elements to skip to get to the next element in each dimension
    char* repr;        // holds the text representation of the tensor
} Tensor;

// Tensor functions
void tensor_setitem(Tensor* t, int* indices, float val);
float tensor_getitem(Tensor* t, int* indices);
void tensor_free(Tensor* t);
void tensor_print(Tensor* t);
Tensor* tensor_empty(int* shape);
Tensor* tensor_arange(float start, float step, int* shape);

#endif