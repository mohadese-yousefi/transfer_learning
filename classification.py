import os
import shutil
import argparse

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory


def train(args):
    train_dir = args['train_directory']
    validation_dir = args['validation_directory']

    # Load data
    train_dataset = image_dataset_from_directory(train_dir,
                                                 shuffle=True,
                                                 batch_size=args['batch_size'],
                                                 image_size=args['img_size'],
                                                 )

    validation_dataset = image_dataset_from_directory(validation_dir,
                                                      shuffle=True,
                                                      batch_size=args['batch_size'],
                                                      image_size=args['img_size'],
                                                      )

    class_names = train_dataset.class_names

    val_batches = tf.data.experimental.cardinality(validation_dataset)
    test_dataset = validation_dataset.take(val_batches // 5)
    validation_dataset = validation_dataset.skip(val_batches // 5)
    '''
    prefetch transformation can be used to decouple the time when data
    is produced from the time when data is consumed. In particular,
    the transformation uses a background thread and an internal buffer
    to prefetch elements from the input dataset ahead of the time
    they are requested. The number of elements to prefetch should be equal to
    (or possibly greater than) the number of batches consumed
    by a single training step. You could either manually tune this value,
    or set it to tf.data.experimental.AUTOTUNE which will prompt the
    tf.data runtime to tune the value dynamically at runtime.
    '''
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    # Augumentation
    data_augmentation = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
      tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
      tf.keras.layers.experimental.preprocessing.RandomZoom(.5, .1),
    ])

    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)

    # Create the base model from the pre-trained model MobileNet V2
    IMG_SHAPE = args['img_size'] + (3,)
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet',
                                                   )

    image_batch, label_batch = next(iter(train_dataset))
    feature_batch = base_model(image_batch)
    print(feature_batch.shape)

    # Freez pretrain model
    base_model.trainable = False

    # Building model
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)
    print(feature_batch_average.shape)

    prediction_layer = tf.keras.layers.Dense(len(class_names))
    prediction_batch = prediction_layer(feature_batch_average)
    print(prediction_batch.shape)

    inputs = tf.keras.Input(shape=(160, 160, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy']
                  )

    loss0, accuracy0 = model.evaluate(validation_dataset)

    print("initial loss: {:.2f}".format(loss0))
    print("initial accuracy: {:.2f}".format(accuracy0))

    history = model.fit(train_dataset,
                        epochs=args['epochs'],
                        validation_data=validation_dataset)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

    #Retrieve a batch of images from the test set
    image_batch, label_batch = test_dataset.as_numpy_iterator().next()
    predictions = model.predict_on_batch(image_batch).flatten()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--class_number",
                    type=int,
                    default="3",
                    help="number of classes",
                    )
    ap.add_argument("-m", "--model_directory",
                    type=str,
                    default="model",
                    help="path to trained model",
                    )
    ap.add_argument("-t", "--train_directory",
                    type=str,
                    help="path to train directory",
                    )
    ap.add_argument("-v", "--validation_directory",
                    type=str,
                    help="path to validation directory",
                    )
    ap.add_argument("-e", "--epochs",
                    type=int,
                    default=10,
                    help="number of ephoch",
                    )
    ap.add_argument("-b", "--batch_size",
                    type=int,
                    default=10,
                    help="size for each batch",
                    )
    ap.add_argument("-s", "--img_size",
                    type=tuple,
                    default=(160, 160),
                    help="size of input image ex:(16, 16)",
                    )
    args = vars(ap.parse_args())

    train(args)

