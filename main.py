#
## Loading Annotations


from Annotations.load_labels import Annotation_to_Numpy, load_pickle
annotation_training = load_pickle(
    'Annotations/annotation_training.pkl')
_, annotation_training = Annotation_to_Numpy(annotation_training)

annotation_validation = load_pickle(
    'Annotations/annotation_validation.pkl')
_, annotation_validation = Annotation_to_Numpy(annotation_validation)

"""## Loading Dataset"""

from DataLoader import DataGenerator
# Dataloader:
Variables = {'num_labels': 5,
             'batch_size': 8,
             'dim': (256, 256),
             'n_channels': 3,
             'shuffle': True,
             'mode': 'torch',
             'dtype': 'float16',
             'reshape_size': 256,
             'number_of_split': 10,
             'number_of_frames': 1,
             'min_neighbors': 10,
             'scalefactor': 1.2
}
path_train = 'Dataset/Train'
training_data = DataGenerator(from_dir=path_train, labels=annotation_training, name='Train', **Variables)

path_val = 'Dataset/Validation'
validation_data = DataGenerator(from_dir=path_val, labels=annotation_validation, name='Val', **Variables)

"""## Building the Model"""

from CustomModel import DisplayNetwork,CustomModel
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.utils.vis_utils import plot_model

model = CustomModel()
model._layers = [layer for layer in model._layers if isinstance(layer, Layer)]
plot_model(model)

print(model.summary())

"""## Compiling the Model

### Callbacks
"""

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import datetime

#Checkpoint Callback
checkpoint_filepath = '/content/drive/MyDrive/First_Impression/Checkpoint/model.{epoch:02d}-{val_loss:.2f}.ckpt'
model_checkpoint_callback = ModelCheckpoint(
                                            filepath=checkpoint_filepath,
                                            save_weights_only=True,
                                            monitor='val_EvalMetric',
                                            mode='auto',
                                            verbose=1,
                                            save_best_only=False)

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                patience=3,
                                verbose=1,
                                factor=0.2,
                                min_lr=5e-7)

# early stopping:
model_earlystopping_callback = EarlyStopping(patience=5)

"""### compile + new_metric"""

import tensorflow.keras.backend as k
import numpy as np
def EvalMetric(y_true,y_pred):
    r2 = 1 - tf.reduce_mean(tf.abs(y_true-y_pred))
    return r2
adam = Adam(lr=1e-4)
model.compile(optimizer=Adam(), loss='mse', metrics=['mae',tf.keras.metrics.CosineSimilarity(), EvalMetric])

"""## Training"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
result = model.fit(training_data, validation_data=validation_data, steps_per_epoch=30, validation_steps=8, epochs=300, callbacks=[model_earlystopping_callback,reduce_lr, model_checkpoint_callback], verbose=1)

a=list(result.history.keys())
a['loss' in a]

from matplotlib import pyplot as plt
for key in list(result.history.keys()):
    plt.plot(result.history[key])
    plt.xlabel("Epoch")
    plt.ylabel(key)
    plt.show()
    print(key)
    
print('The end')
