from tensorflow.keras.layers import Layer, Embedding
from tensorflow import shape, range
from tensorflow.math import not_equal

from character_transformer.utils import ModelSettings

class PositionalEmbedding(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.token_embeddings = Embedding(
            input_dim=len(ModelSettings.VOCABULARY.value), 
            output_dim=ModelSettings.EMBEDDING_DIMENSION.value,
        )
        self.position_embeddings = Embedding(
            input_dim=ModelSettings.SEQUENCE_LENGTH.value, 
            output_dim=ModelSettings.EMBEDDING_DIMENSION.value
        )
        self.sequence_length = ModelSettings.SEQUENCE_LENGTH.value
        self.vocab_size = len(ModelSettings.VOCABULARY.value)
        self.embed_dim = ModelSettings.LATENT_DIMENSION.value

    def call(self, inputs):
        positions = range(start=0, limit=shape(inputs)[-1], delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    @staticmethod
    def compute_mask(inputs, mask=None):
        return not_equal(inputs, 0)