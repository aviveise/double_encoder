from double_encoder import DoubleEncoder
from TrainingStrategy.iterative_training_nonsequential_stratagy import IterativeNonSequentialTrainingStrategy

__author__ = 'aviv'

if __name__ == '__main__':

    DoubleEncoder.run(IterativeNonSequentialTrainingStrategy())