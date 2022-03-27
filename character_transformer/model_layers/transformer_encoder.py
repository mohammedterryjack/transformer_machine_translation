from tensorflow.keras.layers import Layer, MultiHeadAttention, Dense, LayerNormalization
from tensorflow import cast, newaxis
from keras import Sequential

from character_transformer.utils import ModelSettings

class TransformerEncoder(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.embed_dim = ModelSettings.EMBEDDING_DIMENSION.value
        self.dense_dim = ModelSettings.LATENT_DIMENSION.value
        self.num_heads = ModelSettings.NUMBER_HEADS.value
        self.attention = MultiHeadAttention(
            num_heads=ModelSettings.NUMBER_HEADS.value, 
            key_dim=ModelSettings.EMBEDDING_DIMENSION.value
        )
        self.dense_projection = Sequential([
            Dense(ModelSettings.LATENT_DIMENSION.value, activation="relu"), 
            Dense(ModelSettings.EMBEDDING_DIMENSION.value)
        ])
        self.layernorm_1 = LayerNormalization()
        self.layernorm_2 = LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask):
        attended_input = self.attention(
            query=inputs, 
            value=inputs, 
            key=inputs, 
            attention_mask=cast(mask[:, newaxis, newaxis, :], dtype="int32")
        )
        projected_input = self.layernorm_1(inputs + attended_input)
        projected_output = self.dense_projection(projected_input)
        return self.layernorm_2(projected_input + projected_output)