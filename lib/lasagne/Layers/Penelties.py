from theano import tensor


def orthonormality(x):
    return tensor.sum(abs(tensor.dot(x.T, x)))