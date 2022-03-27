from tensorflow.keras.layers import Layer, Embedding
from tensorflow import shape, range
from tensorflow.math import not_equal

from word_transformer.utils import TOKENISER, ModelSettings, EMBEDDINGS

class PositionalEmbedding(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.token_embeddings = Embedding(
            input_dim=2+len(TOKENISER), 
            weights=[EMBEDDINGS],
            output_dim=ModelSettings.EMBEDDING_DIMENSION.value,
            trainable=False
        )
        self.position_embeddings = Embedding(
            input_dim=ModelSettings.SEQUENCE_LENGTH.value, 
            output_dim=ModelSettings.EMBEDDING_DIMENSION.value
        )
        self.sequence_length = ModelSettings.SEQUENCE_LENGTH.value
        self.vocab_size = 2+len(TOKENISER)
        self.embed_dim = ModelSettings.LATENT_DIMENSION.value

    def call(self, inputs):
        positions = range(start=0, limit=shape(inputs)[-1], delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    @staticmethod
    def compute_mask(inputs, mask=None):
        return not_equal(inputs, 0)