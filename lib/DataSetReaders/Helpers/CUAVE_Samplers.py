import sys
import os
import string
import math
import cv2
import cPickle
import numpy
import scipy
import scipy.io.wavfile as wave

from datetime import datetime

__author__ = 'aviv'

FPS_VIDEO = 29.97
FPS_AUDIO = 16100

def extractSpectogram(audio, window_width, window_overlap):

    window = numpy.hamming(window_width)

    count = 0

    specgram = 0

    length = len(audio)

    for i in range(0, length - window_width, window_overlap):

        cur_frame = audio[i: i + window_width]
        window_frame = cur_frame * window
        spec = scipy.fft(window_frame)

        if i == 0:
            width = int(length/window_overlap)
            height = int(spec.size)
            specgram = numpy.empty([height, width])

        for j in range(height):

            val = spec[j]
            specgram[j, count] = math.sqrt(math.pow(val.real, 2) + math.pow(val.imag, 2))

        count += 1

    return specgram


def getVideo(file, start_frames, frame_count, index):

    capture = cv2.VideoCapture(file)

    frames = []

    frame_index = 0
    prev_index = 0

    face_detector = cv2.CascadeClassifier("../../haarcascade_frontalface_default.xml")
    mouth_detector = cv2.CascadeClassifier("../../haarcascade_mouth.xml")

    frames = numpy.ndarray((50, 19200))

    for idx, start_index in enumerate(start_frames[0]):

        for i in range(int((start_index - prev_index) * FPS_VIDEO - frame_count)):
            success, frame = capture.read()

        prev_index = start_index

        for i in range(frame_count):

            success, frame = capture.read()
            if not success:
                print 'reading frame unsuccessful'
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray)

            max_w = 0
            for face in faces:

                if face[2] > max_w:
                    max_w = face[2]
                    max_face = face

            if len(face) == 0:
                raise Exception('no face found')

            height = max_face[3]

            face_gray = gray[max_face[1] + height * 0.2: max_face[1] + height,
                             max_face[0]: max_face[0] + max_face[2]]

            mouths = mouth_detector.detectMultiScale(face_gray, 1.14, 5, 0, minSize=(70, 10))

            max_y = 0
            max_w = 0
            max_mouth = None
            mouth = None
            for mouth in mouths:

                if mouth[1] > max_y:
                    max_y = mouth[1]
                    max_mouth = mouth


            if not mouth is None:

                mouth_gray = face_gray[max_mouth[1]: max_mouth[1] + max_mouth[3], max_mouth[0]: max_mouth[0] + max_mouth[2]]
                cv2.imwrite('/home/aviv/Project/DoubleEncoder/lib/mouths/mouth_%i_%i.jpg' % (index, (idx * frame_count + i + 1)), mouth_gray)
                mouth_gray = cv2.resize(mouth_gray, (80, 60))

            else:
                print 'Could not detect mouth'
                continue

            print 'processed video frame %i of %i' % (idx * frame_count + i + 1, 200)

            frames[idx, i * 4800: (i + 1) * 4800] = mouth_gray.flatten()

    return frames

def getAudio(file, frame_time, frame_count):

    rate, data = wave.read(file)

    frames = numpy.ndarray((50, 4830))
    for i in range(len(frame_time[0])):

        specgrams = extractSpectogram(data[int(frame_time[0][i] * FPS_AUDIO): int(frame_time[1][i] * FPS_AUDIO)],
                                      int(0.02 * FPS_AUDIO),
                                      int(0.01 * FPS_AUDIO))
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

            frame_starts = getFrameStarts(os.path.join(label_path, file_path + ".lab"))
            audio = getAudio(os.path.join(audio_path, file_path + ".wav"), frame_starts, 10)
            video = getVideo(os.path.join(video_path, file_path + ".mpg"), frame_starts, 4, idx)

            if idx % 2 == 0:
                training_audio[(idx / 2) * 50: ((idx / 2) + 1) * 50, :] = audio
                training_video[(idx / 2) * 50: ((idx / 2) + 1) * 50, :] = video

            if not idx % 2 == 0:
                training_audio[((idx - 1) / 2) * 50: (((idx - 1) / 2) + 1) * 50, :] = audio
                training_video[((idx - 1) / 2) * 50: (((idx - 1) / 2) + 1) * 50, :] = video

        except Exception:

            print 'failed processing file'
            print Exception.message




    labels = [i % 10 for i in range(50)]

    cPickle.dumps(training_video, file('./training_video.p', 'w+'))
    cPickle.dumps(training_audio, file('./training_video.p', 'w+'))

    cPickle.dumps(testing_video, file('./testing_video.p', 'w+'))
    cPickle.dumps(testing_audio, file('./testing_audio.p', 'w+'))

    cPickle.dumps(labels, file('./labels.p', 'w+'))