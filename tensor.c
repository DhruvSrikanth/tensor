#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "tensor.h"

// ----------------------------------------------------------------------------
// memory allocation with error checking
void *malloc_check(size_t size, const char *file, int line) {
    void *ptr = malloc(size);
    if (ptr == NULL) {
        fprintf(stderr, "Error: Memory allocation failed at %s:%d\n", file, line);
        exit(EXIT_FAILURE);
    }
    return ptr;
}
#define mallocCheck(size) malloc_check(size, __FILE__, __LINE__)
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// utils
int logical_to_physical(Tensor* t, int* indices) {
    // convert logical indices to physical index
    int idx = t->offset;
    for (int i = 0; i < t->ndim; i++) {
        idx += indices[i] * t->strides[i];
    }
    return idx;
}

// Helper function to recursively fill the string
void fill_string(Tensor* t, int* indices, int dim, char** current) {
    if (dim == t->ndim) {
        // we're at the innermost dimension, add the element
        float value = tensor_getitem(t, indices);
        *current += sprintf(*current, "%.2f", value);
    } else {
        // iterate over the current dimension
        *current += sprintf(*current, "[");
        for (int i = 0; i < t->shape[dim]; i++) {
            indices[dim] = i;
            fill_string(t, indices, dim + 1, current);
            if (i < t->shape[dim] - 1) { *current += sprintf(*current, ", "); }
        }
        *current += sprintf(*current, "]");
    }
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Storage: simple array of floats, defensive on index access, reference-counted
// The reference counting allows multiple Tensors sharing the same Storage.
// similar to torch.Storage
Storage* storage_new(int size) {
    Storage* storage = mallocCheck(sizeof(Storage));
    storage->data = mallocCheck(size * sizeof(float));
    storage->data_size = size;
    storage->ref_count = 1;
    return storage;
}

float storage_getitem(Storage* s, int idx) {
    assert(idx >= 0 && idx < s->data_size);
    return s->data[idx];
}

void storage_setitem(Storage* s, int idx, float val) {
    assert(idx >= 0 && idx < s->data_size);
    s->data[idx] = val;
}

void storage_incref(Storage* s) {
    s->ref_count++;
}

void storage_decref(Storage* s) {
    s->ref_count--;
    if (s->ref_count == 0) {
        free(s->data);
        free(s);
    }
}
// ----------------------------------------------------------------------------
// Tensor: n-dimensional tensor with shape, strides, offset, and reference to Storage
// similar to torch.Tensor
void tensor_setitem(Tensor* t, int* indices, float val) {
    int ndim = sizeof(indices) / sizeof(int);
    if (ndim != t->ndim) {
        fprintf(stderr, "IndexError: Number of indices does not match the number of dimensions\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < t->ndim; i++) {
        // handle negative indices
        if (indices[i] < 0) { indices[i] += t->shape[i]; }
        // check oob
        if (indices[i] < 0 || indices[i] >= t->shape[i]) {
            fprintf(stderr, "IndexError: index %d is out of bounds of %d\n", indices[i], t->shape[i]);
            exit(EXIT_FAILURE);
        }
    }

    int idx = logical_to_physical(t, indices);
    storage_setitem(t->storage, idx, val);
}

float tensor_getitem(Tensor* t, int* indices) {
    int ndim = sizeof(indices) / sizeof(int);
    if (ndim != t->ndim) {
        fprintf(stderr, "IndexError: Number of indices does not match the number of dimensions\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < t->ndim; i++) {
        // handle negative indices
        if (indices[i] < 0) { indices[i] += t->shape[i]; }
        // check oob
        if (indices[i] < 0 || indices[i] >= t->shape[i]) {
            fprintf(stderr, "IndexError: index %d is out of bounds of %d\n", indices[i], t->shape[i]);
            exit(EXIT_FAILURE);
        }
    }

    int idx = logical_to_physical(t, indices);
    return storage_getitem(t->storage, idx);
}

void tensor_free(Tensor* t) {
    storage_decref(t->storage);
    free(t->shape);
    free(t->strides);
    free(t);
}

char* tensor_to_string(Tensor* t) {
    // return a string representation if we already have it
    if (t->repr != NULL) { return t->repr; }
    
    // compute the maximum length of the string representation
    int max_len = 2; // for the outer brackets
    for (int i = 0; i < t->ndim; i++) {
        max_len += t->shape[i] * 5; // 5 chars per element (e.g. -1.23)
        max_len += (t->shape[i] - 1) * 2; // spaces and commas between elements
        if (i < t->ndim - 1) {
            max_len += t->shape[i] * 2; // brackets
            max_len += 2; // spaces and commas between dimensions
        }
    }
    max_len += 1; // null terminator

    // allocate memory for the string representation
    t->repr = mallocCheck(max_len * sizeof(char));
    char* current = t->repr;

    // initialize the current pointer
    current += sprintf(current, "[");

    // allocate an array to hold the indices
    int* indices = (int*) mallocCheck(t->ndim * sizeof(int));
    for (int i = 0; i < t->ndim; i++) { indices[i] = 0; }

    // recursively fill the string
    fill_string(t, indices, 0, &current);

    // finalize the string
    current += sprintf(current, "]");
    current += sprintf(current, "\0");

    // free the indices array
    free(indices);

    return t->repr;
}

void tensor_print(Tensor* t) {
    char* str = tensor_to_string(t);
    printf("%s\n", str);
    free(str);
}

Tensor* tensor_empty(int* shape) {
    // compute the total size required for storage
    int ndim = sizeof(shape) / sizeof(int);
    int size = 1;
    for (int i = 0; i < ndim; i++) { size *= shape[i]; }

    // allocate storage and tensor
    Tensor* t = mallocCheck(sizeof(Tensor));
    t->storage = storage_new(size);
    t->offset = 0;  // offset is 0 when the tensor is created (non-zero for slicing)
    t->ndim = ndim;
    t->shape = mallocCheck(ndim * sizeof(int));
    memcpy(t->shape, shape, ndim * sizeof(int));
    t->strides = mallocCheck(ndim * sizeof(int));
    // strides are computed as the number of elements to skip to get to the next element in each dimension
    t->strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) { t->strides[i] = t->strides[i + 1] * t->shape[i + 1]; }
    t->repr = NULL;
    return t;    
}

// slightly different from the original torch.arange
Tensor* tensor_arange(float start, float step, int* shape) {
    Tensor* t = tensor_empty(shape);
    float val = start;
    for (int i = 0; i < t->storage->data_size; i++) {
        storage_setitem(t->storage, i, val);
        val += step;
    }
    return t;
}
// ----------------------------------------------------------------------------


int main() {
    int shape[] = {3, 4};
    Tensor* t = tensor_arange(0.0, 1.0, shape);
    printf("Tensor:\n");
    tensor_print(t);
    int indices[] = {1, 2};
    float val = tensor_getitem(t, indices);
    printf("Value at index (%d, %d): %.2f\n", indices[0], indices[1], val);
    tensor_free(t);
    return 0;
}