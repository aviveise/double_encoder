import ConfigParser
import sys

import numpy
from sklearn import svm
from lib.MISC.container import Container
from lib.MISC.logger import OutputLog
from lib.MISC.utils import ConfigSectionMap
import lib.DataSetReaders

OUTPUT_DIR = r'C:\Workspace\output'


def concat(x_sample, y_sample):
    #[(vec1 + vec2)/2 ; abs(vec1 - vec2) ; vec1 ; vec2];

    sum = (x_sample + y_sample) / 2
    sub = abs(x_sample - y_sample)
    return numpy.concatenate((sum, sub, x_sample, y_sample))


def concat_pairs(x, y, mapping, reduce):

    x = x[reduce]

    num_x = x.shape[0]
    num_y = y.shape[0]

    labels = numpy.zeros(num_x * num_y)
    pairs = None

    for i, x_sample in enumerate(x):
        for j, y_sample in enumerate(y):
            pair = concat(x_sample, y_sample)
            labels[i * num_y + j] = mapping[i, j]

            if pairs is None:
                pairs = pair
            else:
                pairs = numpy.vstack((pairs, pair))

    return pairs, labels


if __name__ == '__main__':

    data_set_config = sys.argv[1]

    OutputLog().set_path(OUTPUT_DIR)
    OutputLog().set_verbosity('info')

    data_config = ConfigParser.ConfigParser()
    data_config.read(data_set_config)
    data_parameters = ConfigSectionMap("dataset_parameters", data_config)

    # construct data set
    data_set = Container().create(data_parameters['name'], data_parameters)
    data_set.load()

    train_concatenated_pairs, labels = concat_pairs(data_set.trainset[0],data_set.trainset[1], data_set.x_y_mapping['train'], data_set.x_reduce['train'])

    clf = svm.SVC(probability=True)
    clf.fit(train_concatenated_pairs, labels)

    test_concatenated_pairs, labels = concat_pairs(data_set.testset[0],data_set.testset[1], data_set.x_y_mapping['test'], data_set.x_reduce['test'])

    scores = clf.decision_function(test_concatenated_pairs)

    questions = data_set.testset[0][data_set.x_reduce['test']]
    answers_num = data_set.testset[1].shape[0]
    ranks = numpy.zeros(questions)
    aps = numpy.zeros(questions)
    for question_num in range(questions.shape[0]):
        question_pairs_scores = scores[question_num * answers_num: (question_num + 1) * answers_num]
        question_pairs_labels = labels[question_num * answers_num: (question_num + 1) * answers_num]

        pos_scores = numpy.argsort(question_pairs_scores)[::-1][question_pairs_labels]

        r = 1. / float(pos_scores[0])
        ap = numpy.mean(numpy.array([float(i) / float(pos) for i, pos in enumerate(pos_scores)]))

        ranks[question_num] = r
        aps[question_num] = ap


    print 'mrr: {0}, map: {1}'.format(numpy.mean(ranks, numpy.mean(aps)))
