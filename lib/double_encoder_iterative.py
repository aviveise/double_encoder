from lib.double_encoder import DoubleEncoder
from lib.training_strategy.iterative_training_strategy import IterativeTrainingStrategy

__author__ = 'aviv'

if __name__ == '__main__':

    DoubleEncoder.run(IterativeTrainingStrategy())