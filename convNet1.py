from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras import layers, models, backend
from numba import cuda
from imgSrc import learnDir
import os


class convModel:
    def __init__(self):
        self.__model = models.Sequential()
        self.__trainPath = os.path.join(learnDir, 'train', 'proxy')
        self.__testPath = os.path.join(learnDir, 'test')

        self.__train_gen = ImageDataGenerator()
        self.__val_gen = ImageDataGenerator()
        self.__gpu = cuda.get_current_device()

    def addLayers(self, Layers, optimizer='adam', loss='categorical_crossentropy', metrics=None):
        if metrics is None:
            metrics = ['accuracy']
        backend.clear_session()

        for layer in Layers:
            self.__model.add(layer)
        self.__model.compile(optimizer=optimizer,
                             loss=loss,
                             metrics=metrics)

    def trainModel(self, trainPath=None, validationPath=None, color_mode='rgb', target_size=(100, 100), batch_size=30, class_mode='categorical', seed=101, epochs=50):

        if trainPath is None:
            trainPath = self.__trainPath
        if validationPath is None:
            validationPath = self.__testPath

        trainData = self.__train_gen.flow_from_directory(directory=trainPath,
                                                         color_mode=color_mode,
                                                         target_size=target_size,
                                                         batch_size=batch_size,
                                                         class_mode=class_mode,
                                                         seed=seed
                                                         )
        testData = self.__val_gen.flow_from_directory(directory=validationPath,
                                                      color_mode=color_mode,
                                                      target_size=target_size,
                                                      batch_size=batch_size,
                                                      class_mode=class_mode,
                                                      seed=seed
                                                      )

        STEP_SIZE_TRAIN = trainData.n // trainData.batch_size
        STEP_SIZE_VALID = testData.n // testData.batch_size
        self.__model.fit_generator(generator=trainData,
                                   steps_per_epoch=STEP_SIZE_TRAIN,
                                   validation_data=testData,
                                   validation_steps=STEP_SIZE_VALID,
                                   epochs=epochs
                                   )


    def trainExampleModel(self):
        self.__model.add(layers.Conv2D(64, (5, 5), activation='relu', input_shape=(100, 100, 3)))
        self.__model.add(layers.MaxPooling2D((4, 4)))
        self.__model.add(layers.Conv2D(64, (6, 6), activation='relu'))
        self.__model.add(layers.MaxPooling2D((4, 4)))
        self.__model.add(layers.Conv2D(64, (4, 4), activation='relu'))
        self.__model.add(layers.Flatten())
        self.__model.add(layers.Dense(64, activation='relu'))
        self.__model.add(layers.Dense(3))
        self.__model.summary()

        self.__model.compile(optimizer='adam',
                             loss='categorical_crossentropy',
                             metrics=['accuracy'])

        self.trainModel()
        # self.__gpu.reset()

    def saveModelToFile(self):
        self.__model.save(os.path.join(self.__trainPath, 'model'), True)

    def loadModelFromFile(self):
        self.__model = tf.keras.models.load_model(os.path.join(self.__trainPath, 'model'))

    def getJSON(self):
        return self.__model.to_json()

    def getWeights(self):
        return self.__model.get_weights()

    def setJSON(self, json):
        self.__model = models.model_from_json(json)

    def setWeights(self, weights):
        self.__model.set_weights(weights)
