from enum import Enum 
from string import punctuation

class ModelSettings(Enum):
    SEQUENCE_LENGTH = 200
    BATCH_SIZE = 64
    SHUFFLE_SEED = 2048
    PREFETCH = 16
    EMBEDDING_DIMENSION = 256
    LATENT_DIMENSION = 2048
    NUMBER_HEADS = 8
    EPOCHS = 50
    DROPOUT_RATE = 0.2
    MAXIMUM_GENERATION_LENGTH = 100
    ACTIVATION = "relu"
    OPTIMISER = "rmsprop"
    LOSS = "sparse_categorical_crossentropy"
    METRIC = "accuracy"
    MODEL_PATH = "character_transformer/saved/model_weights.hdf5"
    VOCABULARY = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" + punctuation