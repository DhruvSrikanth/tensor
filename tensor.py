import cffi
import numpy as np
# -----------------------------------------------------------------------------
ffi = cffi.FFI()
ffi.cdef("""
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
void tensor_setitem(Tensor* t, int* indices, float val, int ndim);
float tensor_getitem(Tensor* t, int* indices, int ndim);
void tensor_free(Tensor* t);
char* tensor_to_string(Tensor* t);
void tensor_print(Tensor* t);
Tensor* tensor_reshape(Tensor* t, int* shape, int ndim);
Tensor* tensor_empty(int* shape, int ndim);
Tensor* tensor_arange(float start, float step, int* shape, int ndim);
Tensor* tensor_ones(int* shape, int ndim);
Tensor* tensor_zeros(int* shape, int ndim);
""")
lib = ffi.dlopen("./libtensor.so")  # Make sure to compile the C code into a shared library


# -----------------------------------------------------------------------------
# utils
# -----------------------------------------------------------------------------
def _get_shape(data):
    if isinstance(data, (list, np.ndarray)):
        return [len(data)] + _get_shape(data[0])
    else:
        validate_value_type(data)
        return []


def validate_value_type(value):
    if not isinstance(value, (int, float)):
        raise TypeError(f"Invalid value type {value}. Must be int or float.")


def flatten(data):
    if isinstance(data, (int, float)):
        return [data]
    elif isinstance(data, (list, np.ndarray)):
        return [item for sublist in map(flatten, data) for item in sublist]
    else:
        raise TypeError(f"Invalid data type {type(data)}. Must be int, float, list, or numpy array.")


def product(arr):
    result = 1
    for i in arr: result *= i
    return result


# -----------------------------------------------------------------------------
class Tensor:
    def __init__(self, data=None, c_tensor=None):
        # let's ensure only one of data and c_tensor is passed
        assert (data is not None) ^ (c_tensor is not None), "Either data or c_tensor must be passed"
        # let's initialize the tensor
        if c_tensor is not None:
            self.tensor = c_tensor
        elif isinstance(data, (int, float)):
            validate_value_type(data)
            self.tensor = lib.tensor_empty([1], 1)
            lib.tensor_setitem(self.tensor, [0], float(data), 1)
        elif isinstance(data, (list, np.ndarray)):
            shape = _get_shape(data)
            size = product(shape)
            tensor = lib.tensor_empty([size], 1)
            for i, item in enumerate(flatten(data)):
                lib.tensor_setitem(tensor, [i], item, 1)
            self.tensor = lib.tensor_reshape(tensor, shape, len(shape))
        else:
            raise TypeError(f"Data must be int, float, list, or numpy array but got {type(data)}")

    def __del__(self):
        # TODO: when Python intepreter is shutting down, lib can become None
        # I'm not 100% sure how to do cleanup in cffi here properly
        if lib is not None:
            if hasattr(self, 'tensor'):
                lib.tensor_free(self.tensor)

    def __str__(self):
        c_str = lib.tensor_to_string(self.tensor)
        py_str = ffi.string(c_str).decode('utf-8')
        return py_str

    def __repr__(self): return self.__str__()

    def __len__(self): return self.tensor.shape[0]


def tensor(data): return Tensor(data)
