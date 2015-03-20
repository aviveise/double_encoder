import sys
import os
import string
import math
import cv2
import cPickle
import numpy
import scipy
import scipy.io.wavfile as wave
import scipy.io

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

    frames = numpy.ndarray((50, 19200))

    frame_c = 0

    for idx, start_index in enumerate(start_frames[0]):

        for i in range(int((start_index * FPS_VIDEO - frame_c + 1))):
            frame_c += 1
            success, frame = capture.read()

        print 'frame: %i time: %f' % (frame_c, (frame_c / FPS_VIDEO))

        prev_index = start_index
        frame_index = frame_count
        for i in range(frame_count):

            success, frame = capture.read()
            frame_c += 1
            if not success:
                print 'reading frame unsuccessful'
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, minSize=(230, 170))

            max_w = 0
            for face in faces:

                if face[3] > max_w:
                    max_w = face[3]
                    max_face = face

            if len(face) == 0:
                raise Exception('no face found')


            height = max_face[3]

            face_gray = gray[max_face[1] + height * 0.65: max_face[1] + height,
                             max_face[0]: max_face[0] + max_face[2]]

            #cv2.imwrite(os.path.join(mouth_path, 'mouth_' + file_name + '_%i_%i_face.jpg' % (index, (idx * frame_count + i + 1))), face_gray)

            mouths = mouth_detector.detectMultiScale(face_gray, 1.2, 4, 0, minSize=(50, 10))

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
                cv2.imwrite(os.path.join(mouth_path, 'mouth_' + file_name + '_%i_%i.jpg' % (index, (idx * frame_count + i + 1))), mouth_gray)
                mouth_gray = cv2.resize(mouth_gray, (80, 60))

            else:
                print 'Could not detect mouth'
                continue

            print 'processed video frame %i of %i' % (idx * frame_count + i + 1, 200)

            frames[idx, i * 4800: (i + 1) * 4800] = mouth_gray.flatten()

    return frames

def getAudio(file, frame_time, frame_count):

    rate, data = wave.read(file)

    window_width = int(0.02 * FPS_AUDIO)
    window_overlap = int(0.01 * FPS_AUDIO)

    frames = numpy.ndarray((50, 4830))
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

            even_indices = frame[::2]
            odd_indices = frame[1::2]

            frame = numpy.append(frame, odd_indices - even_indices)

            print 'processed audio %i of %i' % (i * frame_count + j + 1, 500)

            frames[i, j * 483: (j + 1) * 483] = frame

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


if __name__ == '__main__':

    video_path = sys.argv[1]
    audio_path = os.path.join(video_path, 'audio')
    label_path = os.path.join(video_path, 'labels')

    files = [f[:-4] for f in os.listdir(video_path) if os.path.isfile(os.path.join(video_path, f))]

    training_video = numpy.ndarray((900, 19200))
    training_audio = numpy.ndarray((900, 4830))

    testing_video = numpy.ndarray((900, 19200))
    testing_audio = numpy.ndarray((900, 4830))

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
                training_audio[(idx / 2) * 50: ((idx / 2) + 1) * 50, :] = audio
                training_video[(idx / 2) * 50: ((idx / 2) + 1) * 50, :] = video

            if not idx % 2 == 0:
                training_audio[((idx - 1) / 2) * 50: (((idx - 1) / 2) + 1) * 50, :] = audio
                training_video[((idx - 1) / 2) * 50: (((idx - 1) / 2) + 1) * 50, :] = video

        except Exception:
            print 'failed processing file ' + file_path
            raise


    labels = [i % 10 for i in range(50)]

    dump = {
        'train_audio': training_audio,
        'train_video': training_video,
        'test_audio': testing_audio,
        'test_video': testing_video,
        'labels': labels
    }

    output = os.path.join(output_path, 'output.mat')
    scipy.io.savemat(output, dump)
