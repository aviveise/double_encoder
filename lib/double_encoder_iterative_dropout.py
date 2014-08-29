from lib.double_encoder import DoubleEncoder
from lib.training_strategy.iterative_dropout_training_strategy import IterativeDropoutTrainingStrategy

__author__ = 'aviv'

if __name__ == '__main__':

    DoubleEncoder.run(IterativeDropoutTrainingStrategy())