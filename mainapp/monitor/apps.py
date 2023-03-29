import os
from django.apps import AppConfig
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow import keras

class ResNetModelConfig(AppConfig):
    name = 'resnetAPI'
    MODEL_FILE = os.path.join(settings.MODELS, "resnet_tumor.h5")
    model = keras.models.load_model(MODEL_FILE)

class VGGModelConfig(AppConfig):
    name = 'vggAPI'
    MODEL_FILE = os.path.join(settings.MODELS, "vgg19_tumor.h5")
    model = keras.models.load_model(MODEL_FILE)

