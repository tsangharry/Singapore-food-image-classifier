import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from polyaxon_client.tracking import get_outputs_path


class FoodClassifier:

    def __init__(self, data_path):
        self.batchsize = 16
        self.train_dir = os.path.join(data_path, 'train')
        self.test_dir = os.path.join(data_path, 'test')
        self.train_datagen = None
        self.test_datagen = None
        self.train_generator = None
        self.validation_generator = None
        self.test_generator = None
        self.model = None
        tf.random.set_seed(16)

    def load_data(self):
        # make sure preprocessing is same as preprocessing as the network
        # reduce mean, and divide by a value to do scaling
        """ Split the data into train, validation, and test"""
        self.train_datagen = ImageDataGenerator(
            rescale=1./ 255,
            shear_range=0.05,
            rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range=[0.9, 1.1],  # Randomly zoom image
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            brightness_range=[0.8, 1.2],
            fill_mode='reflect',
            validation_split=0.2)

        self.test_datagen = ImageDataGenerator(rescale=1. / 255)

        self.train_generator = self.train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(299, 299),
            shuffle=True,
            batch_size=self.batchsize,
            class_mode='categorical',
            subset="training")

        self.validation_generator = self.train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(299, 299),
            shuffle=True,
            batch_size=self.batchsize,
            class_mode='categorical',
            subset="validation")

        self.test_generator = self.test_datagen.flow_from_directory(
            self.test_dir,
            target_size=(299, 299),
            shuffle=False,
            batch_size=1,
            class_mode='categorical')

    def create_model(self, model_base='Xception', base_model_trainable=False, dense_activation=128):
        """Create the neural net with pretrained weights"""
        if model_base == 'Xception':
            model_base = tf.keras.applications.Xception(weights='imagenet', include_top=False,
                                                        input_shape=(299, 299, 3))
        elif model_base == 'InceptionV3':
            model_base = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False,
                                                        input_shape=(299, 299, 3))

        model_base.trainable = base_model_trainable

        x = model_base.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(dense_activation, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        predictions = tf.keras.layers.Dense(12, activation='softmax')(x)

        self.model = tf.keras.Model(inputs=model_base.input, outputs=predictions)

    def train_model(self):
        """Training the model"""
        # callbacks
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=1, verbose=1,
                                                    factor=0.6, min_lr=0.00001)
        checkpointer = ModelCheckpoint('checkpoint.h5', monitor='val_loss', verbose=1, save_best_only=True)
        early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto')

        optimizer = Adam(learning_rate=0.01)

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=['accuracy'])

        history = self.model.fit(self.train_generator,
                                 epochs=20,
                                 shuffle=True,
                                 verbose=1,
                                 validation_data=self.validation_generator,
                                 callbacks=[learning_rate_reduction, checkpointer, early_stopper])

        return history

    def evaluate_model(self):
        """Evaluating the model"""
        model_loss, model_accuracy = self.model.evaluate(self.test_generator)

        return model_loss, model_accuracy

    def save_model(self, directory):
        """Saving the model"""
        os.makedirs(directory)
        h5_directory = get_outputs_path() + '/tensorfood.h5'
        self.model.save(h5_directory)
