import numpy

if __name__ == '__main__':

    samples = numpy.random.multivariate_normal(0, [[1, 0.8], [0.8, 1]], (2, 100))

    numpy.corrcoef(samples[0], samples[1])