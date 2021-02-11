import tensorflow as tf
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import random
import os
import glob
from pyAudioAnalysis import ShortTermFeatures as aF
from pyAudioAnalysis import audioBasicIO as aIO
from sklearn import preprocessing
import matplotlib.pyplot as plt


# %%
class DataGenerator(tf.keras.utils.Sequence):
    """
    Generates data for Keras
    Video and audio frame generator generates batch of frames from a video directory.i.e.:
        videos/file1.mp4
        videos/file2.mp4
        videos/file3.mp4
    """

    # %%
    def __init__(self, from_dir=None, labels=None, num_labels=5, batch_size=8, dim=(256, 256), n_channels=3,
                 shuffle=True, mode='torch', dtype='float16', reshape_size=256, number_of_split=10,
                 number_of_frames=1, min_neighbors=10, scalefactor=1.2, name='train'):
        """
        Create a Video Frame Generator with data augmentation.

        Usage example:
        gen = DatagGenerator('./out/videos/',
            batch_size=8,
            )

        Arguments:
        - from_dir: path to the data directory where resides videos,
            videos should be splitted in directories that are name as labels
        - batch_size: number of videos to generate
        - number_of_frames: number of frames per video partition to send
        - shuffle: boolean, shuffle data at start and after each epoch
        - mode: preprocess and normalize  frames based on different modes. 'torch' normalize features to range (0,1)
        -dtype: dtpye for generated data.
        -number_of_splits: splits input video to splits. Then, pick number_of_frames from each of them.

        -
        """
        # TODO: Change dim to only contains number_of_split, (omit number_of_frames according to previous works)
        # or simply set number_of_frames=1
        self.dim = (number_of_split * number_of_frames, *dim)
        self.name = name
        self.batch_size = batch_size
        self.labels = labels
        self.num_labels = num_labels
        self.from_dir = from_dir
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.mode = mode
        self.dtype = dtype
        self.reshape_size = reshape_size
        self.number_of_split = number_of_split
        self.number_of_frames = number_of_frames
        self.min_neighbors = min_neighbors
        self.scalefactor = scalefactor
        # the list of files, built in __list_all_files
        self.files = []
        self.indexes = []
        # prepare the list
        self.__filecount = 0
        self.__list_all_files()
        self.on_epoch_end()

    # %%
    def __len__(self):
        """ Length of the generator
        Warning: it gives the number of loop to do, not the number of files or
        frames. The result is number_of_video/batch_size. You can use it as
        `step_per_epoch` or `validation_step` for `model.fit_generator` parameters.
        """
        return int(np.floor(len(self.files) / self.batch_size))

    # %% File location
    def __list_all_files(self):
        """
        List all files in directory
        """
        self.files = glob.glob(os.path.join(self.from_dir, '*.mp4'))
        self.__filecount = len(self.files)
        self.indexes = np.arange(len(self.files))

        if self.shuffle:
            random.shuffle(self.files)

    # %%
    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        '"""
        self.indexes = np.arange(len(self.files))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    # %%
    def __getitem__(self, index):
        """
        Generator needed method - return a batch of `batch_size` video
        block with `self.dim[0]` for each
        this method provides batch of data for machine learning model
        """
        # TODO: index should be fixed
        try:
            # index = np.random.randint(0, len(self.files) - 1 - self.batch_size)
            # Generate indexes of the batch
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
            # Find list of IDs
            from_dir_temp = [self.files[k] for k in indexes]
        except Exception as e:
            print(e)
            print('error in loading the batch')
            from_dir_temp = self.files
        # Generate data
        X, y = self.__data_generation(from_dir_temp)
        print('Data is generated, shape=', X[0].shape)
        return X, y

    # %% Face Detection
    def get_face(self, image):
        """
        This function find the face position in the input image
        """
        test_image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        haar_cascade_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        condition = True
        min_neighbors = self.min_neighbors
        while condition:
            faces_rects = haar_cascade_face.detectMultiScale(test_image_gray, scaleFactor=self.scalefactor,
                                                             minNeighbors=min_neighbors)
            if len(faces_rects) == 0 and min_neighbors >= 2:
                min_neighbors -= 2
            elif len(faces_rects) == 0 and min_neighbors < 2:
                return 0
            else:
                condition = False
                # print('Faces found: ', len(faces_rects))
                return faces_rects
        # print('Faces found: ', len(faces_rects))

    # %%
    def __data_generation(self, from_dir_temp):
        """
        Generates data containing batch_size samples.
    Parameters:
        -------
        from_dir_temp: location of the chozen files from the directory.
        :return
        inputs: [data_tf, audio_tf] where data_tf is concatenated data from video files, and audio_tf is extracted
        audio features
        y: labels
        """
        # TODO: change the shape of X: (batch_size, splits*frames,256,256,3) or (batch_size*splits*frames,256,256,3)
        # X : (n_samples, *dim, n_channels)
        # Initialization
        # TODO: fix X_audio final dimension (68). It shouldn't be a predefined number.
        # TODO: fix X_audio reshape formula
        X_image = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float16)
        X_audio = np.empty((self.batch_size, self.dim[0], 68))
        y = np.empty((self.batch_size, self.dim[0], self.num_labels), dtype=np.float16)

        print('Loading Batch has started',self.name)
        # Generate data, len(from_dir_temp) = batch_size
        for i, ID in enumerate(from_dir_temp):
            # Load and store sample
            video = VideoFileClip(ID)
            video_length = video.reader.nframes  # total number of frames
            # Select some of the frames from each video
            FrameIndices = self.RandomGenerator(video_length, IsSequential=False)
            VideoData = self.VideoSampling(video, FrameIndices)
            AudioVector = self.AudioFeatures(video, FrameIndices)
            AudioVector = AudioVector.transpose()
            X_image[i, ] = VideoData
            X_audio[i, ] = AudioVector
            try:
                # Store labels
                id_file = ID.split('/')[-1]
                # y[i*self.batch_size:(i+1)*self.batch_size, :] = self.labels[id_file]
                y[i, :] = self.labels[id_file]
            except Exception as e:
                print(e)
                print(ID)
                pass

        # reshape = (-1, *self.dim[1:], self.n_channels)    reshape to (batches*samples,dim)
        # X_image = np.reshape(X_image, reshape)
        data_tf = tf.convert_to_tensor(X_image, dtype=tf.float16)

        # X_audio = np.reshape(X_audio, (-1, 68))
        audio_tf = tf.convert_to_tensor(X_audio, dtype=tf.float16)
        inputs = [data_tf, audio_tf]
        y = tf.convert_to_tensor(y, dtype=tf.float16)
        print('Loading has finished', self.name)
        return inputs, y
        # for categorical labels: keras.utils.to_categorical(y, num_classes=self.n_classes)

    # %% Video Sampling
    def VideoSampling(self, video, frames):
        """
        This function randomly select number_of_frames*number_of_split from the video
        number_of_split divides video frames to n split.
        """
        # Face Detection
        # choose a random frame of the video to get face coordinates, doesn't matter which one
        faces_rects = self.get_face(video.get_frame(0))
        # choose only one of the faces, in case of extracting multiple faces
        try:
            face_rect = faces_rects[0]
            # face coordinates:
            # (x,y) left-top corner
            # (w,h) width and height of the face
            (x, y, w, h) = face_rect
        except:
            # in case faces_rect is zero, choose a random pixel from the video
            (Height, Width) = video.reader.size
            (r1, r2) = (np.random.randint(Width - self.reshape_size), np.random.randint(Height - self.reshape_size))
            (x, y, w, h) = (r1, r2, self.dim[1], self.dim[2])
        # preprocessing: normalizing the dataset
        sample = self.Preprocessor(video, frames, (x, y, w, h))
        return sample

    # %% Randomly select frames
    def RandomGenerator(self, video_length, IsSequential=True):
        """

        :param video_length: total number of video frames
        :param IsSequential: boolean. If false, #number of frames from each partition of the video
        :return: frame_indices. Randomly selected frames from each video file in the batch
        """
        frame_range = int(video_length / self.number_of_split)
        ranges = [(i * frame_range, (i + 1) * frame_range - self.number_of_frames) for i in
                  np.arange(self.number_of_split)]
        if IsSequential:
            frame_list = []
            # rand = lambda x, y: np.random.randint(x, y)
            frame_list.extend([np.random.randint(start, stop) for start, stop in ranges])
            frames = [list(range(i, i + self.number_of_frames)) for i in frame_list]
        else:
            frames = [random.sample(range(*ranges[i]), self.number_of_frames) for i in range(len(ranges))]
        # concatenate all frames
        frame_indices = []
        for frame in frames:
            frame_indices.extend(frame)
        frame_indices = sorted(frame_indices)
        return frame_indices

    # %% Audio Features
    def AudioFeatures(self, video, frame_indices):
        """
        :param video: input data of type video
        :param frame_indices: randomly selected frames to extract their audio features
        :return: extracted features for each video file. A normalized vector of length 68
        """
        audio = video.audio
        signal = audio.to_soundarray()  # array
        # change dtype of signal
        signal = signal.astype(np.float16)
        sig = aIO.stereo_to_mono(signal)
        # partition steps:
        val = audio.duration / (self.number_of_split * self.number_of_frames)
        win = step = val
        # sampling frequency:
        fs = int(sig.shape[0] / audio.duration)
        # extracting features from each partition. returns a matrix of the shape (68, #splits)
        [f1, _] = aF.feature_extraction(sig, fs, int(fs * win), int(fs * step))
        # normalization
        #f1 = preprocessing.StandardScaler().fit(f1).transform(f1.astype(np.float16))
        f1 = preprocessing.normalize(f1, axis=0)
        return f1

    # %%
    def Preprocessor(self, video, frame_indices, indexes):
        """

        :param video: input file before being processed
        :param frame_indices: frames generated from RandomGenerator method, list
        :param indexes: face position in the image
        :return: normalized frames of video
        """
        (x, y, w, h) = indexes
        Samples = np.empty((*self.dim, self.n_channels), dtype=np.uint8)
        fps = video.fps
        (Height, Width) = video.reader.size
        Cropped_video = video.crop(x1=x, y1=y, width=400, height=400)
        try:
            Cropped_video = Cropped_video.resize(self.dim[1:])
        except:
            Cropped_video = video.crop(x1=x, y1=y, width=self.dim[1], height=self.dim[2])
        else:
            if Cropped_video.size != self.dim[1:]:
                Cropped_video = video.crop(x1=1, y1=1, width=self.dim[1], height=self.dim[2])
        # Threshold: 1/3 of the frames are randomly cropped to add more diversity in features
        threshold = int(2 / 3 * len(frame_indices))
        for i, frame_num in enumerate(frame_indices):
            if i < threshold:
                try:
                    Samples[i, :] = Cropped_video.get_frame(frame_num / fps)
                except:
                    if i > 1:
                        Samples[i, :] = Samples[:i, :].mean(axis=0)
                    else:
                        Samples[i, :] = np.random.random(size=(*self.dim[1:],self.n_channels)) * 255
            else:
                # Crop a random window from the video clip
                (r1, r2) = (np.random.randint(Width - self.reshape_size), np.random.randint(Height - self.reshape_size))
                Samples[i, :] = video.get_frame(frame_num / fps)[r1:r1 + self.reshape_size, r2: r2 + self.reshape_size,
                                :]
            # plt.imshow(Samples[i, :])
            # plt.show()
            # resize frames for training
            # reshape_size = (self.reshape_size, self.reshape_size, 3)
            # Samples[i, :] = np.resize(sample, reshape_size)

        # preprocess (normalize and scale) data
        # torch: normalize w.r.t imagenet to range [0,1] for features, 'tf' mode: normalize w.r.t imagenet to range
        # [-1,1], 'caffe':
        Samples = preprocess_input(Samples, mode=self.mode)
        Samples = Samples.astype(self.dtype)
        return Samples


# In[Driver]
# from Annotations.load_labels import Annotation_to_Numpy, load_pickle
#
# # #
# annotation_training = load_pickle('/home/alikazemi/PycharmProjects/pythonProject/Annotations'
#                                   '/annotation_training.pkl')
# _, annotation_training = Annotation_to_Numpy(annotation_training)
# path = '/home/alikazemi/PycharmProjects/pythonProject/Dataset/Training'
# obj = DataGenerator(from_dir=path, labels=annotation_training,batch_size=20)
# X, y = obj.getitem(1)
# # #FileNames = os.listdir(path)

# from_dir_temp = [path + files for files in FileNames if 'mp4' in files]
# XX = obj.data_generation(from_dir_temp)
# In[Original Driver]
# params = {'batch_size':64, 'dim':(48,48), 'n_classes':2, 'is_autoencoder':True, 'shuffle':True }
# train_gen = DataGenerator(path_to_traindata,**params)
# validn_gen = DataGenerator(path_to_validationdata,**params)
