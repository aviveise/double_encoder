""" This file contains different utility functions that are not connected
in anyway to the networks presented in the tutorials, but rather help in
processing the outputs into a more understandable way.

For example ``tile_raster_images`` helps in generating a easy to grasp
image from a set of samples or weights.
"""

import math
import os
import datetime
from matplotlib import pyplot
from matplotlib.pyplot import pcolor, colorbar, yticks, xticks, pcolormesh, matplotlib
import numpy
import numpy.linalg
import scipy.linalg
import scipy.sparse.linalg
from sklearn import preprocessing
from MISC.logger import OutputLog

global file_ndx
file_ndx = {}

matplotlib.pyplot.ioff()


def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)_
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`PIL.Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                 in zip(img_shape, tile_shape, tile_spacing)]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)

        # colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = numpy.zeros(out_shape,
                                                 dtype=dt) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = numpy.zeros(out_shape, dtype=dt)

        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                    tile_row * (H + Hs): tile_row * (H + Hs) + H,
                    tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
        return out_array


def unitnorm_rows(M):
    if M is None:
        return

    for i in xrange(M.shape[0]):

        norm = numpy.linalg.norm(M[i, :])
        if norm != 0:
            M[i, :] /= norm

    return M


def unitnorm_cols(M):
    if M is None:
        return

    for i in xrange(M.shape[1]):

        norm = numpy.linalg.norm(M[:, i])
        if norm != 0:
            M[:, i] /= norm

    return M


def normalize(M):
    if M is None:
        return

    norm = numpy.linalg.norm(M, ord=2, axis=1).reshape([M.shape[0], 1])

    norm[norm == 0] = 1

    M /= norm * numpy.ones([1, M.shape[1]])

    return M, norm


def scale_cols(M):
    # scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(M)
    scaler = preprocessing.StandardScaler().fit(M)
    return scaler.transform(M), scaler


def center(M):
    if M is None:
        return

    mean = M.mean(axis=1).reshape([M.shape[0], 1])
    M -= mean * numpy.ones([1, M.shape[1]])
    return M, mean


def print_list(list, percentage=False):
    if percentage:
        list = [i * 100 for i in list]
        return '[%s%%]' % '%,'.join(map(str, list))
    else:
        return '[%s]' % ','.join(map(str, list))


def find_correlation(x, y, svd_sum=True):
    forward = unitnorm_rows(center(x))
    backward = unitnorm_rows(center(y))

    if svd_sum:

        return numpy.linalg.svd(numpy.dot(forward, backward.T), compute_uv=False).sum()

    else:

        correlation = numpy.ndarray([forward.shape[0], 1])

        for i in xrange(forward.shape[0]):
            correlation[i] = numpy.corrcoef(forward, backward)[0, 1]

        return correlation.sum()


def cca_representation(x, y, eta=0, apply_r=True):
    mean_x1 = x.mean(axis=1)
    mean_x2 = y.mean(axis=1)

    # reshape to transform the array to ndarray
    mean_x1 = mean_x1.reshape([x.shape[0], 1])
    mean_x2 = mean_x2.reshape([y.shape[0], 1])

    x -= numpy.dot(mean_x1, numpy.ones([1, x.shape[1]], numpy.float))
    y -= numpy.dot(mean_x2, numpy.ones([1, y.shape[1]], numpy.float))

    if eta:
        wx, wy, r = cca_web2(x, y, regfactor=(1 / eta))
    else:
        wx, wy, r = cca_web2(x, y, regfactor=0)

    xi = numpy.dot(wx[:, 0:50].T, x)
    yi = numpy.dot(wy[:, 0:50].T, y)

    if apply_r:
        xi = numpy.dot(numpy.diag(r), xi)
        yi = numpy.dot(numpy.diag(r), yi)

    return xi, yi


def cca_web2(x, y, xi=None, regfactor=0, regfactor2=0):
    #
    # CCA calculate canonical correlations
    #
    # [Wx Wy r] = cca(X,Y) where Wx and Wy contains the canonical correlation
    # vectors as columns and r is a vector with corresponding canonical
    # correlations. The correlations are sorted in descending order. X and Y
    # are matrices where each column is a sample. Hence, X and Y must have
    # the same number of columns.
    #
    # Example: If X is M*K and Y is N*K there are L=MIN(M,N) solutions. Wx is
    # then M*L, Wy is N*L and r is L*1.
    #
    #
    # modified by lw to make symmetric and add regularization
    #

    if not xi:
        xi = numpy.ones(x.shape[0])
    else:
        xi = xi / numpy.linalg.norm(xi) * numpy.sqrt(x.shape[0])

    if not regfactor2:
        regfactor2 = 10 ** (-8)

    z = numpy.concatenate([x, y])

    covariance = numpy.cov(z)
    size_x = x.shape[0]
    size_y = y.shape[0]

    c_xx = covariance[0:size_x, 0:size_x] + regfactor2 * numpy.eye(size_x)

    if regfactor != 0:
        c_xx += numpy.diag(xi) * (scipy.sparse.linalg.eigs(c_xx, k=1, which='LM')[0] / regfactor)

    c_xy = covariance[0:size_x, size_x:size_x + size_y]
    c_yx = c_xy.conj().transpose()

    c_yy = covariance[size_x:size_x + size_y, size_x:size_x + size_y] + regfactor2 * numpy.eye(size_y)

    if regfactor != 0:
        c_yy += numpy.eye(size_y) * (scipy.sparse.linalg.eigs(c_yy, k=1, which='LM')[0] / regfactor)

    inv_c_xx = numpy.linalg.inv(c_xx)
    inv_c_yy = numpy.linalg.inv(c_yy)

    d = min(size_x, size_y)
    mx = numpy.dot(numpy.dot(numpy.dot(inv_c_xx, c_xy), inv_c_yy), c_yx)

    if d < size_x:
        r, wx = scipy.sparse.linalg.eigs(mx, k=d, which='LM')
    else:
        r, wx = numpy.linalg.eig(mx)

    r = numpy.sqrt(numpy.real(r))

    v = numpy.fliplr(wx)
    r = numpy.flipud(r)
    i = numpy.argsort(numpy.real(r), axis=0)
    r = numpy.sort(numpy.real(r), axis=0)
    r = numpy.flipud(r)

    for j in xrange(i.shape[0]):
        wx[:, j] = v[:, i[j]]

    wx = numpy.fliplr(wx)

    my = numpy.dot(numpy.dot(numpy.dot(inv_c_yy, c_yx), inv_c_xx), c_xy)

    if d < size_y:
        r, wy = scipy.sparse.linalg.eigs(my, k=d, which='LM')
    else:
        r, wy = numpy.linalg.eig(my)

    r = numpy.sqrt(numpy.real(r))

    v = numpy.fliplr(wy)
    r = numpy.flipud(r)
    i = numpy.argsort(numpy.real(r), axis=0)
    r = numpy.sort(numpy.real(r), axis=0)
    r = numpy.flipud(r)

    for j in xrange(i.shape[0]):
        wy[:, j] = v[:, i[j]]

    wy = numpy.fliplr(wy)

    for i in xrange(wx.shape[1]):
        wx[:, i] = wx[:, i] / numpy.sqrt(numpy.dot(wx[:, i].T, numpy.dot(c_xx, wx[:, i])))

    for i in xrange(wy.shape[1]):
        wy[:, i] = wy[:, i] / numpy.sqrt(numpy.dot(wy[:, i].T, numpy.dot(c_yy, wy[:, i])))

    xx = numpy.real(numpy.dot(wx.conj().T, x))
    yy = numpy.real(numpy.dot(wy.conj().T, y))

    signs = numpy.ndarray(xx.shape[0])
    for i in xrange(xx.shape[0]):

        if i < yy.shape[0] and numpy.linalg.norm(xx[i, :]) * numpy.linalg.norm(yy[i, :]):
            signs[i] = numpy.sign(numpy.correlate(xx[i, :].conj().T, yy[i, :].conj().T))
        else:
            signs[i] = 1

    wy = numpy.dot(wy, numpy.diag(signs))

    return wx, wy, r


def ConfigSectionMap(section, config):
    dict1 = {}

    try:
        options = config.options(section)

    except:
        return None

    for option in options:
        try:
            dict1[option] = config.get(section, option)

        except:
            dict1[option] = None

    return dict1


def testWhitenTransform(data):
    colNum = data.shape[1]
    rowNum = data.shape[0]
    mu = numpy.dot(data, numpy.ones([colNum, 1])) * (1 / float(colNum))
    print 'mu error: %f' % numpy.linalg.norm(mu)

    sigma = numpy.dot(data, data.T) * (1 / float(colNum - 1))

    for i in xrange(rowNum):
        sigma[i, i] -= 1

    print 'sigma error: %f' % numpy.linalg.norm(sigma)


def convertInt2Bitarray(number):
    number_length = int(math.ceil(math.log(number, 2)))
    result = numpy.ndarray([1, number_length])
    for i in xrange(number_length):
        result[0, i] = number % 2
        number = number >> 1

    return result


def calculate_square(x):
    w, v = numpy.linalg.eigh(x)

    n = x.shape[0]
    result = numpy.zeros(x.shape)

    for i in xrange(x.shape[0]):
        result += numpy.sqrt(w[i]) * numpy.dot(v[:, i].reshape(n, 1), v[:, i].reshape(1, n))

    return result


def match_error(x, y, visualize):
    x_c = center(x)[0]
    y_c = center(y)[0]

    x_n = preprocessing.normalize(x_c, axis=1)
    y_n = preprocessing.normalize(y_c, axis=1)

    sym = numpy.dot(x_n, y_n.T)

    if visualize:
        visualize_correlation_matrix(sym, 'similarity_mat')

    top_1 = numpy.argmax(sym, axis=0)
    error = 1 - float(numpy.sum(top_1 == range(x.shape[0]))) / x.shape[0]

    return error


def calculate_mardia(x, y, top, visualize):
    set_size = x.shape[0]
    dim = x.shape[1]

    x, mean_x = center(x.T)
    y, mean_y = center(y.T)

    # correlation_matrix = numpy.corrcoef(x, y)

    s11 = numpy.diag(numpy.diag(numpy.dot(x, x.T) / (set_size - 1) + 10 ** (-8) * numpy.eye(dim, dim)))
    s22 = numpy.diag(numpy.diag(numpy.dot(y, y.T) / (set_size - 1) + 10 ** (-8) * numpy.eye(dim, dim)))
    s12 = numpy.dot(x, y.T) / (set_size - 1)

    s11_chol = scipy.linalg.sqrtm(s11)
    s22_chol = scipy.linalg.sqrtm(s22)

    s11_chol_inv = scipy.linalg.inv(s11_chol)
    s22_chol_inv = scipy.linalg.inv(s22_chol)

    mat_T = numpy.dot(numpy.dot(s11_chol_inv, s12), s22_chol_inv)

    # mat_T = correlation_matrix[0:x.shape[0], x.shape[0]: x.shape[0] + y.shape[0]]

    if visualize:
        visualize_correlation_matrix(mat_T, 'correlation_mat')
        visualize_correlation_matrix(numpy.sort(mat_T, axis=1), 'correlation_mat_sorted')

    s = numpy.linalg.svd(numpy.diag(numpy.diag(mat_T)), compute_uv=0)

    del mat_T

    if top == 0:
        return numpy.sum(s)

    return numpy.sum(s[0:top])


def calculate_trace(x, y):
    centered_x = center(x)
    centered_y = center(y)

    forward = unitnorm_rows(centered_x)
    backward = unitnorm_rows(centered_y)

    diagonal = numpy.abs(numpy.diagonal(numpy.dot(forward, backward.T)))
    diagonal.sort()
    diagonal = diagonal[::-1]

    return sum(diagonal)


def calculate_corrcoef(x, y, top):
    n = x.shape[0]
    corr = numpy.corrcoef(x, y)

    corr = corr[0: n, n + 1: 2 * n]

    diag = numpy.abs(numpy.diagonal(corr))
    diag.sort()
    diag = diag[::-1]

    return numpy.sum(diag[0:top])


def calculate_reconstruction_error(x, y):
    return numpy.mean(((x - y) ** 2).sum(axis=1))


def complete_rank(x, y, reduce_x=0):
    x_c = center(x)[0]
    y_c = center(y)[0]

    x_n = preprocessing.normalize(x_c, axis=1)
    y_n = preprocessing.normalize(y_c, axis=1)

    num_X_samples = x.shape[0]
    num_Y_samples = y.shape[0]

    if reduce_x:
        x_n = x_n[0:x_n.shape[0]:reduce_x, :]
        y_x_mapping = numpy.repeat(numpy.arange(x_n.shape[0]), reduce_x)
    else:
        y_x_mapping = numpy.arange(x_n.shape[0])

    y_x_sim_matrix = numpy.dot(x_n, y_n.T)

    recall_n_vals = [1, 5, 10]
    num_of_recall_n_vals = len(recall_n_vals)

    x_search_recall = numpy.zeros((num_of_recall_n_vals, 1))
    describe_x_recall = numpy.zeros((num_of_recall_n_vals, 1))

    x_search_sorted_neighbs = numpy.argsort(y_x_sim_matrix, axis=0)[::-1, :]
    x_search_ranks = numpy.array(
        [numpy.where(col == y_x_mapping[index])[0] for index, col in enumerate(x_search_sorted_neighbs.T)])

    for idx, recall in enumerate(recall_n_vals):
        x_search_recall[idx] = numpy.sum(x_search_ranks <= recall)

    x_search_recall = 100 * x_search_recall / num_Y_samples

    describe_y_sorted_neighbs = numpy.argsort(y_x_sim_matrix, axis=1)[:, ::-1]
    describe_y_ranks = numpy.array([numpy.where(numpy.in1d(row, numpy.where(y_x_mapping == index)[0]))[0]
                                    for index, row in enumerate(describe_y_sorted_neighbs)])

    for idx, recall in enumerate(recall_n_vals):
        describe_x_recall[idx] = numpy.sum(describe_y_ranks.min(axis=0) <= recall)

    describe_x_recall = 100 * describe_x_recall / num_X_samples

    return x_search_recall, describe_x_recall


def visualize_correlation_matrix(mat, name):
    path = OutputLog().output_path
    output_file = os.path.join(path, name + '.jpg')

    if name not in file_ndx:
        file_ndx[name] = 0

    if os.path.exists(output_file):
        output_file = os.path.join(path, name + '_' + str(file_ndx[name]) + '.jpg')
        file_ndx[name] += 1

    f = pyplot.figure()
    pcolormesh(mat)
    colorbar()
    f.savefig(output_file, format='jpeg')
    f.clf()
    pyplot.close()


def calc_correlation_matrix(x, y):
    set_size = x.shape[0]
    dim = x.shape[1]

    x, mean_x = center(x.T)
    y, mean_y = center(y.T)

    s11 = numpy.diag(numpy.diag(numpy.dot(x, x.T) / (set_size - 1) + 10 ** (-8) * numpy.eye(dim, dim)))
    s22 = numpy.diag(numpy.diag(numpy.dot(y, y.T) / (set_size - 1) + 10 ** (-8) * numpy.eye(dim, dim)))
    s12 = numpy.dot(x, y.T) / (set_size - 1)

    s11_chol = scipy.linalg.sqrtm(s11)
    s22_chol = scipy.linalg.sqrtm(s22)

    s11_chol_inv = scipy.linalg.inv(s11_chol)
    s22_chol_inv = scipy.linalg.inv(s22_chol)

    mat_T = numpy.dot(numpy.dot(s11_chol_inv, s12), s22_chol_inv)

    return mat_T
