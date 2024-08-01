import itertools

def ndindex(shape):
    return itertools.product(*[range(dim) for dim in shape])