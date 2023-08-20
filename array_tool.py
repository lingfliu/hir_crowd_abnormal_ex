import numpy as np
def to_categorical(input, num_classes):
    """convert input label into categorical vector"""
    # transform the last dim of input into class vectors
    cate_input = []
    for i in input:
        vec = [0]*num_classes
        vec[int(i)] = 1
        cate_input.append(vec)
    return np.array(cate_input)

