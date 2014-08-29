import sys

from lib.configuration import CrossEncoderParameters

from lib.DataSetReaders.dataset_factory import DatasetFactory

__author__ = 'aviv'


if __name__ == '__main__':
    path = sys.argv[1]

    parameters = CrossEncoderParameters(path)
    data_factory = DatasetFactory()

    args = (parameters.dataset_path,
            parameters.center,
            parameters.normalize,
            parameters.whiten)

    dataset = data_factory.create(parameters.data_type, *args)

    dataset.export_dat()
