from enum import Enum 

from ffast import load 
from ffast.poincare.utils import VECTORS
from numpy import concatenate, zeros

TOKENISER = load('poincare')

class ModelSettings(Enum):
    SEQUENCE_LENGTH = 20
    BATCH_SIZE = 64
    SHUFFLE_SEED = 2048
    PREFETCH = 16
    EMBEDDING_DIMENSION = 100
    LATENT_DIMENSION = 2048
    NUMBER_HEADS = 8
    EPOCHS = 50
    DROPOUT_RATE = 0.5
    MAXIMUM_GENERATION_LENGTH = 20
    ACTIVATION = "relu"
    OPTIMISER = "rmsprop"
    LOSS = "sparse_categorical_crossentropy"
    METRIC = "accuracy"
    MODEL_PATH = "word_transformer/saved/model_weights_poincare.hdf5"
    UNK_ID = len(TOKENISER)+1

EMBEDDINGS = concatenate([VECTORS,zeros(shape=(2,ModelSettings.EMBEDDING_DIMENSION.value))])