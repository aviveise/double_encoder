import os

import hickle
import numpy

DIR = 'C:\Theses\double_encoder\Datasets\GUY'

if __name__ == '__main__':
    train_file = os.path.join(DIR, 'train.p')
    out_file = os.path.join(DIR, 'train_out.p')

    trainset = None
    with open(train_file, 'r') as f:
        trainset = hickle.load(f)

    trainset = (trainset[0][0:trainset[0].shape[0]:5],
                numpy.array([numpy.mean(trainset[1][i * 5: (i + 1) * 5], axis=0) for i in range(trainset[1].shape[0] / 5)]))

    with open(out_file, 'w') as f:
        hickle.dump(trainset, out_file)