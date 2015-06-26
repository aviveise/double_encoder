from double_encoder import DoubleEncoder
from TrainingStrategy.iterative_training_strategy import IterativeTrainingStrategy
from TrainingStrategy.mirror_training_strategy import MirrorTrainingStrategy

__author__ = 'aviv'

if __name__ == '__main__':

    DoubleEncoder.run(MirrorTrainingStrategy())