import tarfile
import numpy

from PIL import Image
from theano import config

from dataset_base import DatasetBase
from MISC.container import ContainerRegisterMetaClass

TRAINING_PERCENT = 0.8

class CARSDataSetTar(DatasetBase):

    __metaclass__ = ContainerRegisterMetaClass

    def __init__(self, data_set_parameters):
        super(CARSDataSetTar, self).__init__(data_set_parameters)

    def build_dataset(self):

        cars_tar = tarfile.open(self.dataset_path)

        files = cars_tar.getmembers()
        cars = {}

        for file_info in files:

            file_name = file_info.name

            if not file_name.endswith('.jpg'):
                continue

            splitted_file_name = file_name.rsplit('_')

            if splitted_file_name[0].endswith('w'):
                car_photo = {'car_id': int(splitted_file_name[2]), 'camera': int(splitted_file_name[3]), 'file_info': file_info}
            if splitted_file_name[0].endswith('e'):
                car_photo = {'car_id': int(splitted_file_name[2]) + 1000, 'camera': int(splitted_file_name[3]), 'file_info': file_info}

            if not car_photo['car_id'] in cars.keys():
                cars[car_photo['car_id']] = {}
                cars[car_photo['car_id']][1] = []
                cars[car_photo['car_id']][2] = []

            cars[car_photo['car_id']][car_photo['camera']].append(car_photo)

        train_pair_number = 0
        tune_pair_number = 0
        test_pair_number = 0
        car_num = 0
        for car in cars.values():

            if car_num < len(cars.values()) * TRAINING_PERCENT * 0.9:
                train_pair_number += min(len(car[1]), len(car[2]))

            elif car_num < len(cars.values()) * TRAINING_PERCENT:
                tune_pair_number += min(len(car[1]), len(car[2]))

            else:
                test_pair_number += min(len(car[1]), len(car[2]))

            car_num += 1

        self.trainset = [numpy.ndarray([1200, train_pair_number], dtype=config.floatX),
                         numpy.ndarray([1200, train_pair_number], dtype=config.floatX)]

        self.tuning = [numpy.ndarray([1200, tune_pair_number], dtype=config.floatX),
                       numpy.ndarray([1200, tune_pair_number], dtype=config.floatX)]

        self.testset = [numpy.ndarray([1200, test_pair_number], dtype=config.floatX),
                        numpy.ndarray([1200, test_pair_number], dtype=config.floatX)]

        pair_number = 0
        for car in cars.values():
            for i in xrange(min(len(car[1]), len(car[2]))):
                image_x1 = Image.open(cars_tar.extractfile(car[1][i]['file_info']), 'r')
                image_x2 = Image.open(cars_tar.extractfile(car[2][i]['file_info']), 'r')

                image_x1 = self.preprocess(image_x1)
                image_x2 = self.preprocess(image_x2)

                if pair_number < train_pair_number:
                    self.trainset[0][:, pair_number] = numpy.asarray(image_x1).reshape([1200])
                    self.trainset[1][:, pair_number] = numpy.asarray(image_x2).reshape([1200])

                elif pair_number < train_pair_number + tune_pair_number:
                    self.tuning[0][:, pair_number - train_pair_number] = numpy.asarray(image_x1).reshape([1200])
                    self.tuning[1][:, pair_number - train_pair_number] = numpy.asarray(image_x2).reshape([1200])

                else:
                    self.testset[0][:, pair_number - train_pair_number - tune_pair_number] = numpy.asarray(image_x1).reshape([1200])
                    self.testset[1][:, pair_number - train_pair_number - tune_pair_number] = numpy.asarray(image_x2).reshape([1200])

                pair_number += 1

    def preprocess(self, image):

        return_image = image.resize([20, 20])

        return return_image


