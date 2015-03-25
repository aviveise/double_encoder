import sys
import os
import string
import math
import cv2
import h5py
import cPickle
import hickle
import numpy
import scipy
import scipy.io.wavfile as wave
import scipy.io

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from datetime import datetime

__author__ = 'aviv'

FPS_VIDEO = 29.97
FPS_AUDIO = 16100

def extractSpectogram(audio, window_width, window_overlap):

    window = numpy.hamming(window_width)

    count = 0

    specgram = 0

    length = len(audio)

    print 'length: %i' % length

    for i in range(0, length - window_width, window_overlap):

        cur_frame = audio[i: i + window_width]
        window_frame = cur_frame * window
        spec = scipy.fft(window_frame)

        if i == 0:
            width = int(length/window_overlap)
            height = int(spec.size)
            print 'width: %i, height: %i' % (width, height)
            specgram = numpy.empty([height, width])

        for j in range(height):

            val = spec[j]
            specgram[j, count] = math.sqrt(math.pow(val.real, 2) + math.pow(val.imag, 2))

        count += 1

    return specgram


def getVideo(file, start_frames, frame_count, index, file_name):

    capture = cv2.VideoCapture(file)

    frames = []

    frame_index = 0
    prev_index = 0

    face_detector = cv2.CascadeClassifier("../../haarcascade_frontalface_default.xml")
    mouth_detector = cv2.CascadeClassifier("../../haarcascade_mouth.xml")
    mouth_path = '/home/aviveise/double_encoder/lib/DataSetReaders/Helpers/mouths/' + file_name[0:3].upper()

    frames = numpy.ndarray((200, 4800))

    frame_c = 0

    for idx, start_index in enumerate(start_frames[0]):

        for i in range(int((start_index * FPS_VIDEO - frame_c + 1))):
            frame_c += 1
            success, frame = capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, minSize=(190, 140))

        max_w = 0
        for face in faces:

            if face[3] > max_w:
                max_w = face[3]
                max_face = face

        if len(face) == 0:
            raise Exception('no face found')

        height = max_face[3]

        prev_index = start_index
        frame_index = frame_count
        for i in range(frame_count):
            success, frame = capture.read()
            frame_c += 1
            if not success:
                print 'reading frame unsuccessful'
                break

            path = os.path.join(mouth_path, 'mouth_' + file_name + '_%i_%i.jpg' % (index, (idx * frame_count + i + 1)))

            if os.path.exists(path):
                mouth_gray = cv2.imread(path)
                mouth_gray = cv2.cvtColor(mouth_gray, cv2.COLOR_BGR2GRAY)
                mouth_gray = cv2.resize(mouth_gray, (80, 60))

            else:

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_gray = gray[max_face[1] + height * 0.55: max_face[1] + height * 1.05,
                                 max_face[0]: max_face[0] + max_face[2]]

                mouths = mouth_detector.detectMultiScale(face_gray, 1.2, 4, 0, minSize=(50, 40))

                max_y = 0
                max_w = 0
                max_mouth = None
                mouth = None
                for mouth in mouths:

                    if mouth[1] > max_y:
                        max_y = mouth[1]
                        max_mouth = mouth


                if mouth is not None:

                    mouth_gray = face_gray[max_mouth[1]: max_mouth[1] + max_mouth[3], max_mouth[0]: max_mouth[0] + max_mouth[2]]
                    cv2.imwrite(path, mouth_gray)
                    mouth_gray = cv2.resize(mouth_gray, (80, 60))

                else:
                    print 'Could not detect mouth'
                    continue

            print 'processed video frame %i of %i' % (idx * frame_count + i + 1, 200)

            frames[idx * frame_count + i, :] = mouth_gray.flatten()

    return frames

def getAudio(file, frame_time, frame_count):

    rate, data = wave.read(file)

    window_width = int(0.02 * FPS_AUDIO)
    window_overlap = int(0.01 * FPS_AUDIO)

    frames = numpy.ndarray((500, 322))
    for i in range(len(frame_time[0])):

        frame_start = int(frame_time[0][i] * FPS_AUDIO)
        frame_end = int(frame_time[1][i] * FPS_AUDIO)

        if ((frame_end - frame_start) - window_width) / window_overlap < frame_count:
            frame_end +=  frame_count * window_overlap + window_width - (frame_end + frame_start)

        specgrams = extractSpectogram(data[frame_start: frame_end],
                                      window_width,
                                      window_overlap)
        for j in range(frame_count):

            frame = specgrams[:, j]

            assert not numpy.any(numpy.isnan(frame) | numpy.isinf(frame))

            print 'processed audio %i of %i' % (i * frame_count + j + 1, 500)

            frames[i * frame_count + j, :] = frame

    return frames

def getFrameStarts(file):

    with open(file) as file:
        content = file.readlines()

    output_start = []
    output_end = []

    for line in content:

        if line == '# Moving\r\n':
            break

        if not line or line[0] == '#' or line == '\r\n':
            continue

        words = string.split(line,' ')

        start_time = datetime.strptime(words[0], '%M:%S.%f')
        end_time = datetime.strptime(words[1], '%M:%S.%f')

        output_start += [start_time.minute * 60 + start_time.second + start_time.microsecond * 0.000001]
        output_end += [end_time.minute * 60 + end_time.second + end_time.microsecond * 0.000001]

    return output_start, output_end

def add_first_second_deriviatives(set):

    #normalize set
    normalized_set = normalize(set, axis=1)

    first_deriviative = numpy.ndarray(set.shape)
    second_deriviative = numpy.ndarray(set.shape)

    t = numpy.arange(set.shape[1])
    dt = numpy.gradient(t)


    for i in range(set.shape[0]):
        first_deriviative[i, :] = numpy.gradient(normalized_set[i, :], dt)
        second_deriviative[i, :] = numpy.gradient(first_deriviative[i ,:], dt)

    return numpy.concatenate((set, first_deriviative, second_deriviative), axis=1)

def group_frames(set, group_size):

    grouped_set = numpy.ndarray((set.shape[0] / group_size, set.shape[1]))
    line_size = set.shape[1]

    for i in range(grouped_set.shape[0]):
        for j in range(group_size):
            grouped_set[i, j * line_size : (j + i) * line_size] = set[j * group_size + i, :]

    return grouped_set

if __name__ == '__main__':

    video_path = sys.argv[1]
    audio_path = os.path.join(video_path, 'audio')
    label_path = os.path.join(video_path, 'labels')

    files = [f[:-4] for f in os.listdir(video_path) if os.path.isfile(os.path.join(video_path, f))]

    training_video = numpy.ndarray((900, 19200))
    training_audio = numpy.ndarray((900, 9660))

    testing_video = numpy.ndarray((900, 19200))
    testing_audio = numpy.ndarray((900, 9660))

    video_frames_training = numpy.ndarray((3600, 4800))
    video_frames_testing = numpy.ndarray((3600, 4800))

    audio_frames_training = numpy.ndarray((9000, 322))
    audio_frames_testing = numpy.ndarray((9000, 322))

    for idx, file_path in enumerate(files):

        print 'processing file: ' + file_path

        try:

            output_path = '/home/aviveise/double_encoder/DataSet/CUAVE/'

            pickle_path = os.path.join(output_path + 'pickle', file_path + '.mat')

            if os.path.exists(pickle_path):
                mat_file = scipy.io.loadmat(pickle_path)
                audio = mat_file['audio']
                video = mat_file['video']
                frame_starts = mat_file['frame_starts']

            else:

                frame_starts = getFrameStarts(os.path.join(label_path, file_path + ".LAB"))
                audio = getAudio(os.path.join(audio_path, file_path + ".WAV"), frame_starts, 10)
                video = getVideo(os.path.join(video_path, file_path + ".MPG"), frame_starts, 4, idx, file_path)

                pickle_file = {
                    'frame_starts': frame_starts,
                    'audio': audio,
                    'video': video
                }

                scipy.io.savemat(pickle_path, pickle_file)

            if idx % 2 == 0:
                audio_frames_training[(idx / 2) * 500: ((idx / 2) + 1) * 500, :] = audio
                video_frames_training[(idx / 2) * 200: ((idx / 2) + 1) * 200, :] = video

            if not idx % 2 == 0:
                audio_frames_testing[((idx - 1) / 2) * 500: (((idx - 1) / 2) + 1) * 500, :] = audio
                video_frames_testing[((idx - 1) / 2) * 200: (((idx - 1) / 2) + 1) * 200, :] = video

        except Exception:
            print 'failed processing file ' + file_path
            raise

    print 'audio frames shape - test [%i, %i], train [%i, %i]' % (audio_frames_testing.shape[0],
                                                                  audio_frames_testing.shape[1],
                                                                  audio_frames_training.shape[0],
                                                                  audio_frames_training.shape[1])

    print 'video frames shape - test [%i, %i], train [%i, %i]' % (video_frames_testing.shape[0],
                                                                  video_frames_testing.shape[1],
                                                                  video_frames_training.shape[0],
                                                                  video_frames_training.shape[1])

    #audio_frames_training = add_first_second_deriviatives(audio_frames_training)
    #audio_frames_testing = add_first_second_deriviatives(audio_frames_testing)

    print 'audio with deriviatives:'
    print 'audio frames shape - test [%i, %i], train [%i, %i]' % (audio_frames_testing.shape[0],
                                                                  audio_frames_testing.shape[1],
                                                                  audio_frames_training.shape[0],
                                                                  audio_frames_training.shape[1])

    pca_video = PCA(n_components=96, whiten=True)
    pca_audio = PCA(n_components=100, whiten=True)

    audio_frames_training = pca_audio.fit_transform(audio_frames_training)
    video_frames_training = pca_video.fit_transform(video_frames_training)

    audio_frames_testing = pca_audio.transform(audio_frames_testing)
    video_frames_testing = pca_video.transform(video_frames_testing)

    print 'audio & video after pca:'
    print 'audio frames shape - test [%i, %i], train [%i, %i]' % (audio_frames_testing.shape[0],
                                                                  audio_frames_testing.shape[1],
                                                                  audio_frames_training.shape[0],
                                                                  audio_frames_training.shape[1])

    print 'video frames shape - test [%i, %i], train [%i, %i]' % (video_frames_testing.shape[0],
                                                                  video_frames_testing.shape[1],
                                                                  video_frames_training.shape[0],
                                                                  video_frames_training.shape[1])


    #video_frames_training= add_first_second_deriviatives(video_frames_training)
    #video_frames_testing = add_first_second_deriviatives(video_frames_testing)

    print 'video with deriviatives:'
    print 'video frames shape - test [%i, %i], train [%i, %i]' % (video_frames_testing.shape[0],
                                                                  video_frames_testing.shape[1],
                                                                  video_frames_training.shape[0],
                                                                  video_frames_training.shape[1])

    print 'video with grouped:'


    video_frames_training  = group_frames(video_frames_training, 4)
    video_frames_testing = group_frames(video_frames_testing, 4)

    audio_frames_training = group_frames(audio_frames_training, 10)
    audio_frames_testing = group_frames(audio_frames_testing, 10)

    print 'audio frames shape - test [%i, %i], train [%i, %i]' % (audio_frames_testing.shape[0],
                                                                  audio_frames_testing.shape[1],
                                                                  audio_frames_training.shape[0],
                                                                  audio_frames_training.shape[1])

    print 'video frames shape - test [%i, %i], train [%i, %i]' % (video_frames_testing.shape[0],
                                                                  video_frames_testing.shape[1],
                                                                  video_frames_training.shape[0],
                                                                  video_frames_training.shape[1])



    dump = {
        'train_audio': audio_frames_training,
        'train_video': video_frames_training,
        'test_audio': audio_frames_testing,
        'test_video': video_frames_testing,
    }

    output = os.path.join(output_path, 'output.hkl')
    hickle.dump(dump, output, 'w')
    hickle.load(output)
