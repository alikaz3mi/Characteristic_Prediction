import tensorflow as tf
import numpy as np
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Flatten, LSTM, Dropout, Concatenate, AveragePooling1D,\
    TimeDistributed
from tensorflow.keras.layers import Layer


def DisplayNetwork(MyModel):
    """
    This functions gets a model and plots its graph
    :param MyModel: built-in model
    :return: graph of the model
    """
    MyModel._layers = [layer for layer in MyModel._layers if isinstance(layer, Layer)]
    plot_model(MyModel)
    return


def CustomModel(input_shape_video=(256, 256, 3),
                input_shape_audio=(10, 68),
                Timestep=10,
                dense_resnet_shape=[1024, 128],
                dense_lstm_shape=5,
                activations_res=['relu', 'linear'],
                activation_audio='linear',
                activation_LSTM='sigmoid',
                audioFeature_shape=32,
                lstm_units=10,
                dropout_prob=.2
                ):
    input_a = Input(shape=(Timestep,*input_shape_video), name='Images')
    input_b = Input(shape=input_shape_audio, name='Audio Features')
    res = tf.keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_shape=input_shape_video)
    # convolution layer ara non-trainable:
    for layer in res.layers:
        layer.trainable = False

    # image features:
    a = TimeDistributed(res, input_shape=(Timestep, *input_shape_video))(input_a)
    a = TimeDistributed(tf.keras.layers.Flatten())(a)
    a = TimeDistributed(Dense(dense_resnet_shape[0], activation=activations_res[0]))(a)
    a = TimeDistributed(Dense(dense_resnet_shape[1], activation=activations_res[1]))(a)
    output_a = TimeDistributed(Dropout(dropout_prob))(a)

    # audio features:
    output_b = TimeDistributed(Dense(audioFeature_shape, activation=activation_audio))(input_b)
    # Concatenate features
    features = Concatenate()([output_a, output_b])

    # LSTM
    x_lstm = LSTM(units=lstm_units, name='lstm',  return_sequences=True)(features)
    x_lstm = Dense(units=dense_lstm_shape, activation=activation_LSTM)(x_lstm)
    output = AveragePooling1D(pool_size=Timestep)(x_lstm)
    print(output.shape)
    model_concat = Model(inputs=[input_a, input_b], outputs=output)
    return model_concat
#%%
# a = CustomModel()
# class CustomModels(tf.keras.Model):
#     def __init__(self,
#                  input_shape,
#                  dense1_shape=512,
#                  dense2_shape=128,
#                  audioFeature_shape=32,
#                  lstm_units=10,
#                  lstm_input_shape=(1, 128)):
#         # The number of filters and kernel_size can be parameterized in here
#         super(CustomModel, self).__init__(name='')
#         self.input_shape = input_shape
#         self.dense1_shape = dense1_shape
#         self.dense2_shape = dense2_shape
#         self.audioFeature_shape = audioFeature_shape
#
#         # resnet required architectures
#         self.__resnet()
#         # audio network
#         self.audioDense = tf.keras.layers.Dense(lstm_units, activation='relu')
#
#         self.concat = tf.keras.layers.Concatenate()
#         # LSTM
#         self.lstm = tf.keras.layers.LSTM(units=lstm_units, input_shape=lstm_input_shape, return_sequences=True)
#
#
#         self.act = tf.keras.layers.Activation('relu')
#         self.add = tf.keras.layers.Add()
#
#     def __resnet(self):
#         self.res = tf.keras.applications.resnet.ResNet50(include_top=False, weights='imagenet',
#         input_shape=self.input_shape)
#         for layer in self.res.layers[-3]:
#             layer.trainable = False
#         self.flatten = tf.keras.layers.Flatten()
#         self.dense1 = tf.keras.layers.Dense(self.dense1_shape, activation='relu')
#         self.dense2 = tf.keras.layers.Dense(self.dense2_shape, activation='relu')
#         self.dropout = tf.keras.layers.Dropout(rate=.2)
#
#     def call(self, inputs):
#         """
#
#         :param inputs: output from DataLoader which generates X_image, X_audio, and y
#         :return: output of the model
#         """
#         input_image, input_audio, _ = inputs
#
#         # resnet model
#         x_res = self.res(input_image)
#         x_res = self.flatten(x_res)
#         x_res = self.dense1(x_res)
#         x_res = self.dense2(x_res)
#         res_output = self.dropout(x_res)
#
#         # audio model
#         audio_output = self.audioDense(input_audio)
#
#         # lstm
#         lstm_input = self.concat([res_output, audio_output])
#         x_lstm = self.lstm(lstm_input)
#
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.act(x)
#
#         x = self.add([x, input_image])
#         x = self.act(x)
#         return x
