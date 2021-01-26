import tensorflow as tf

physical_devices = tf.config.list_physical_devices()
print(physical_devices)
tf.config.set_visible_devices([], 'GPU')

print(tf.config.get_visible_devices())

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import layers, models, backend
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from sklearn.metrics import confusion_matrix
import numpy as np
from numba import cuda
from imgSrc import learnDir, __main as main, covid_classes
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageFilter



class convModel:
    def __init__(self, trainPath='proxy'):
        self.__name = trainPath
        self.__model = models.Sequential()
        self.__trainPath = os.path.join(learnDir, 'train', trainPath)
        self.__testPath = os.path.join(learnDir, 'test')
        self.__modelPath = os.path.join(main, 'models')
        self.__modelType = ''

        self.__history = None
        self.__train_gen = ImageDataGenerator()
        self.__val_gen = ImageDataGenerator()
        self.__gpu = cuda.get_current_device()
        self.__batchSize = 64
        self.__seed = 101
        self.__xSize = 224
        self.__ySize = 224

    def getPaths(self):
        return 'train: ' + self.__trainPath + ', test: ' + self.__testPath + ', save: ' + self.__modelPath

    def __compileModel(self, optimizer='adam', loss='categorical_crossentropy', metrics=None):
        print('compiling')
        if metrics is None:
            metrics = ['accuracy']

        self.__model.compile(optimizer=optimizer,
                             loss=loss,
                             metrics=metrics)

    def addLayers(self, Layers, optimizer='adam', loss='categorical_crossentropy', metrics=None):
        backend.clear_session()

        for layer in Layers:
            self.__model.add(layer)
        self.__compileModel()

    def getAccuracy(self):
        valid_datagen = ImageDataGenerator()
        valid_generator = valid_datagen.flow_from_directory(
            directory=self.__testPath,
            target_size=(self.__xSize, self.__ySize),
            color_mode="rgb",
            batch_size=self.__batchSize,
            class_mode="categorical",
            shuffle=True,
            seed=42
        )
        return self.__model.evaluate_generator(valid_generator)

    def getConfusionMatrix(self, savePath=None, nameAddition=None):
        valid_datagen = ImageDataGenerator()
        valid_generator = valid_datagen.flow_from_directory(
            directory=self.__testPath,
            target_size=(self.__xSize, self.__ySize),
            color_mode="rgb",
            batch_size=self.__batchSize,
            class_mode="categorical",
            shuffle=True,
            seed=42
        )
        Y_pred = self.__model.predict_generator(valid_generator)
        y_pred = np.argmax(Y_pred, axis=1)

        matrix = confusion_matrix(valid_generator.classes, y_pred)
        classes = []
        for c in covid_classes:
            classes.append(c[1:])

        ax = plt.subplot()
        sns.heatmap(matrix, annot=True, ax=ax, cmap='Oranges', annot_kws={"size": 16})
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title(self.__name + ' confusion matrix')
        ax.xaxis.set_ticklabels(classes)
        ax.yaxis.set_ticklabels(classes)
        ax.tick_params(axis='y', rotation=45)
        if savePath is None:
            plt.show()
        else:
            path = os.path.join(savePath, self.__name, self.__name + "_" + self.__modelType + '_confusion_matrix.png')
            if nameAddition:
                path = os.path.join(savePath, self.__name, self.__name + "_" + self.__modelType + '_' + nameAddition + '_confusion_matrix.png')
            plt.savefig(path)
            plt.close()

    def trainModel(self, trainPath=None, validationPath=None, epohs=10):

        if trainPath is None:
            trainPath = self.__trainPath
        if validationPath is None:
            validationPath = self.__testPath

        train_datagen = ImageDataGenerator()
        valid_datagen = ImageDataGenerator()

        train_generator = train_datagen.flow_from_directory(
            directory=trainPath,
            target_size=(self.__xSize, self.__ySize),
            color_mode="rgb",
            batch_size=self.__batchSize,
            class_mode="categorical",
            shuffle=True,
            seed=42
        )

        valid_generator = valid_datagen.flow_from_directory(
            directory=validationPath,
            target_size=(self.__xSize, self.__ySize),
            color_mode="rgb",
            batch_size=self.__batchSize,
            class_mode="categorical",
            shuffle=True,
            seed=42
        )

        STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
        STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size
        self.__history = self.__model.fit_generator(generator=train_generator,
                                                    steps_per_epoch=STEP_SIZE_TRAIN,
                                                    validation_data=valid_generator,
                                                    validation_steps=STEP_SIZE_VALID,
                                                    epochs=epohs
                                                    )

    def __addTop(self, x, nclasses=3):

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(4000, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(1000, activation='relu')(x)
        return layers.Dense(nclasses, activation='relu')(x)

    def vggNet(self, nclasses=3, summary=True):

        vgg = VGG16(weights='imagenet', include_top=False, input_shape=(self.__xSize, self.__ySize, 3))
        vgg.trainable = False
        input_l = layers.Input(shape=(self.__xSize, self.__ySize, 3))
        x = vgg(input_l, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(4000, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(1000, activation='relu')(x)
        predict = layers.Dense(nclasses, activation='relu')(x)
        self.__model = tf.keras.Model(input_l, predict)

        if summary:
            self.__model.summary()

        self.__model.compile(optimizer='adam',
                             loss='categorical_crossentropy',
                             metrics=['accuracy'])

    def resNet(self, nclasses=3, summary=True):
        res = ResNet50V2(weights='imagenet', include_top=False, input_shape=(self.__xSize, self.__ySize, 3))
        res.trainable = False
        input_l = layers.Input(shape=(self.__xSize, self.__ySize, 3))
        x = res(input_l, training=False)
        predict = self.__addTop(x, nclasses)
        self.__model = tf.keras.Model(input_l, predict)
        if summary:
            self.__model.summary()

        self.__compileModel()

    def inception(self, nclasses=3, summary=True):
        inc = InceptionV3(weights='imagenet', include_top=False, input_shape=(self.__xSize, self.__ySize, 3))
        inc.trainable = False
        input_l = layers.Input(shape=(self.__xSize, self.__ySize, 3))
        x = inc(input_l, training=False)
        predict = self.__addTop(x, nclasses)
        self.__model = tf.keras.Model(input_l, predict)
        if summary:
            self.__model.summary()

        self.__compileModel()

    def saveModelToFile(self, name=''):
        self.__model.save(os.path.join(self.__modelPath, name), overwrite=True, save_format='tf')

    def loadModelFromFile(self, name='', summary=True):
        print('Loading model')
        self.__model = tf.keras.models.load_model(os.path.join(self.__modelPath, name))
        self.__unsetTrainable()
        if summary:
            self.__model.summary()
        self.__compileModel()
        if name == 'vgg':
            self.__modelType = 'vggNet'
        elif name == 'inc':
            self.__modelType = 'InceptionNet'
        elif name == 'res':
            self.__modelType = 'ResNet'

    def __unsetTrainable(self):
        self.__model.layers[1].trainable = False

    def getJSON(self):
        return self.__model.to_json()

    def getWeights(self):
        return self.__model.get_weights()

    def getTrainableWeights(self):
        return self.__model.trainable_weights

    def setJSON(self, json):
        """

        @:param json str: JSON string with layer structure of class neural net

        """
        self.__model = models.model_from_json(json)

    def setWeights(self, weights):
        """

        @:param weights np.array: array o new weights to be applied to model.
        """
        self.__model.set_weights(weights)
        self.__compileModel()

    def setTrainableWeights(self, weights):
        """@setTrainableWeights
            @:param weights np.array: numpy array or list of tensor variables. These are only trainable weights.
            """
        non_trainable = self.__model.non_trainable_weights
        all_weights = []
        for i in non_trainable:
            all_weights.append(i.numpy())
        if type(weights).__module__ == np.__name__:
            for i in weights:
                all_weights.append(i)
        else:
            for i in weights:
                all_weights.append(i.numpy())

        self.__model.set_weights(np.array(all_weights))

    def learningCurves(self, savePath=None):
        """@learningCurves

        @:param savePath str: learning curves save path. If save path is None then plot is displayed.
        """
        plt.title(self.__name + ' learning curves')
        plt.xlabel('Epoch')
        plt.ylabel('Cross Entropy')
        plt.plot(self.__history.history['loss'], label='train')
        plt.plot(self.__history.history['val_loss'], label='val')
        plt.legend()
        if savePath is None:
            plt.show()
        elif savePath is not None:
            plt.savefig(os.path.join(savePath, self.__name, self.__name + '_learning_curves.png'))
            plt.close()

    def setNet(self, netType, summary=True):
        """@setNet

        Method that create transfer learning net structure for class instance neural Net.

        @:param netType str: neural net type.
            Currently supported:
            'vgg' - VGG Net,
            'res' - ResNet,
            'inc' - Inception Net
        :param summary:
        :return:
        """
        if netType == 'vgg':
            self.vggNet(summary=summary)
        elif netType == 'inc':
            self.inception(summary=summary)
        elif netType == 'res':
            self.resNet(summary=summary)
