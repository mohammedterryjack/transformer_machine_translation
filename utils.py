from enum import Enum 

from ffast import load 

TOKENISER = load('poincare')

class ModelSettings(Enum):
    SEQUENCE_LENGTH = 20
    BATCH_SIZE = 64
    SHUFFLE_SEED = 2048
    PREFETCH = 16
    EMBEDDING_DIMENSION = 100
    LATENT_DIMENSION = 2048
    NUMBER_HEADS = 8
    EPOCHS = 1000
    DROPOUT_RATE = 0.5
    MAXIMUM_GENERATION_LENGTH = 20
    ACTIVATION = "relu"
    OPTIMISER = "rmsprop"
    LOSS = "sparse_categorical_crossentropy"
    METRIC = "accuracy"
    MODEL_PATH = "conditional_generator_custom/saved/model_weights_poincare.hdf5"
    UNK_ID = len(TOKENISER)+1