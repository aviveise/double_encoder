import os
import sys
import ConfigParser
import scipy
import datetime

from numpy.random import RandomState

from time import clock

from configuration import Configuration

from Testers.trace_correlation_tester import TraceCorrelationTester

from Transformers.double_encoder_transformer import DoubleEncoderTransformer
from Transformers.gradient_transformer import GradientTransformer
from double_encoder import DoubleEncoder
from TrainingStrategy.iterative_training_strategy import IterativeTrainingStrategy

from stacked_double_encoder import StackedDoubleEncoder

from MISC.container import Container
from MISC.utils import ConfigSectionMap
from MISC.logger import OutputLog

from Testers.trace_correlation_tester import TraceCorrelationTester

from Transformers.double_encoder_transformer import DoubleEncoderTransformer

import DataSetReaders
import Regularizations
import Optimizations

__author__ = 'aviv'

if __name__ == '__main__':
    DoubleEncoder.run(IterativeTrainingStrategy(), False)
