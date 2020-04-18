import tensorflow as tf

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, Dropout

import os
import numpy as np
import matplotlib.pyplot as plt

from data_loader import load_samples, concat_two_output_samples, generator


# Data preparation

PATH = "data"
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'val')

train_head_dir = os.path.join(train_dir, 'data_head_pose')  # directory with our training cat pictures
train_body_dir = os.path.join(train_dir, 'data_body_pose')  # directory with our training dog pictures
validation_head_dir = os.path.join(validation_dir, 'data_head_pose')  # directory with our validation cat pictures
validation_body_dir = os.path.join(validation_dir, 'data_body_pose')

train_head_sample = load_samples(train_head_dir)
train_body_sample = load_samples(train_body_dir)
validation_head_sample = load_samples(validation_head_dir)
validation_body_sample = load_samples(validation_body_dir)

train_samples = concat_two_output_samples(train_head_sample, train_body_sample)
val_samples = concat_two_output_samples(validation_head_sample, validation_body_sample)

train_batch_size = 32
val_batch_size = 32

train_generator = generator(train_samples, batch_size=train_batch_size, shuffle_data=True)
val_generator = generator(val_samples, batch_size=val_batch_size, shuffle_data=False)

INPUT_SIZE = 32
HIDDEN_SIZE_1 = 25
HIDDEN_SIZE_2 = 25
OUTPUT_SIZE = 7
# Model init
model = Sequential([
    BatchNormalization(),
    Dense(HIDDEN_SIZE_1, input_shape=(INPUT_SIZE, ), activation='relu'),
    Dropout(0.15),
    Dense(HIDDEN_SIZE_2, activation='relu'),
    Dense(OUTPUT_SIZE),
])

model.build([None, INPUT_SIZE])
model.summary()
model.compile(
            optimizer=tf.python.keras.optimizers.Adam(),
            loss='categorical_crossentropy',
            metrics=['acc'])

# Train procedure
steps_per_epoch = np.ceil(len(train_samples)/train_batch_size)
val_steps_per_epoch = np.ceil(len(val_samples)/val_batch_size)

hist = model.fit(
                train_generator,
                epochs=10,
                verbose=1,
                steps_per_epoch=steps_per_epoch,
                validation_data=val_generator,
                validation_steps=val_steps_per_epoch).history

model.save("best_classifier_model.h5", save_format='h5')



