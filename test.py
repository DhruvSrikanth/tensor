import numpy as np
import tensor

if __name__ == '__main__':
    arr = [[1, 2, 3], [4, 5, 6]]
    arr = np.array(arr)
    t = tensor.tensor(arr)
    print(t)
