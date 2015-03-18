from classifier import Classifier
from TrainingStrategy.iterative_training_strategy import IterativeTrainingStrategy


__author__ = 'aviv'

if __name__ == '__main__':

    Classifier.run(IterativeTrainingStrategy())