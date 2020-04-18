import os
import numpy as np
from random import shuffle


ClassName2Label = {
    'looking_around': 0,
    'lying_on_desc': 1,
    'touching_face': 2,
    'using_mobile': 3,
    'watching_attentively': 4,
    'writing_down_notes': 5,
    'uncertain': 6
}


def load_samples(path_data_head_pose=''):
    samples = []
    for person in os.listdir(path_data_head_pose):
        for class_name in os.listdir(os.path.join(path_data_head_pose, person)):
            files_dir = os.path.join(path_data_head_pose, person, class_name)
            for file_name in os.listdir(files_dir):
                samples.append([os.path.join(files_dir, file_name), ClassName2Label[class_name]])
    return samples


# Usage:
# samples = load_samples('data/train/data_head_pose/')


def concat_two_output_samples(sample_head, sample_body):
    if len(sample_body) != len(sample_head):
        print("sample lenghts are different!")
        return 0
    sample = []
    for i in range(len(sample_head)):
        head_path = sample_head[i][0]
        head_label = sample_head[i][1]
        body_path = sample_body[i][0]
        body_label = sample_body[i][0]

        head_path_name = os.path.splitext(os.path.basename(head_path))[0]
        body_path_name = os.path.splitext(os.path.basename(body_path))[0]
        # TODO: match data path
        if head_label == body_label and head_path_name == body_path_name:
            sample.append([head_path, body_path, head_label])

    if len(sample) == len(sample_head):
        print("Everything matched perfectly!")
    else:
        print("Some data has been lost!")
        print(len(sample))
    return sample


def generator(samples, batch_size=32, shuffle_data=True):
    """
    Yields the next training batch.
    Suppose `samples` is an array [[image1_filename,label1], [image2_filename,label2],...].
    """
    num_samples = len(samples)
    while True:  # Loop forever so the generator never terminates
        if shuffle_data:
            shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            # Get the samples you'll use in this batch
            batch_samples = samples[offset:offset + batch_size]
            # Initialise X_train and y_train arrays for this batch
            X_train = []
            y_train = []

            # For each example
            for batch_sample in batch_samples:
                # Load image (X) and label (y)
                head_pose_name = batch_sample[0]
                body_pose_name = batch_sample[1]
                label = batch_sample[2]

                # TODO: Read numpy archives and append it into X_train
                numpy_input = np.zeros(48)

                X_train.append(numpy_input)
                y_train.append(label)

            # Make sure they're numpy arrays (as opposed to lists)
            X_train = np.array(X_train)
            y_train = np.array(y_train)

            # The generator-y part: yield the next training batch
            yield X_train, y_train


# Import list of train and validation data (image filenames and image labels)
# train_samples = load_samples(flower_recognition_train.csv)
# validation_samples = load_samples(flower_recognition_test.csv)

# # Create generator
# train_generator = generator(train_samples, batch_size=32)
# validation_generator = generator(validation_samples, batch_size=32)