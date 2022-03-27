from tensorflow.keras.layers import Layer, MultiHeadAttention, Dense, LayerNormalization
from tensorflow import cast, newaxis, shape, range, minimum, reshape, concat, tile, expand_dims, constant
from keras import Sequential

from character_transformer.utils import ModelSettings

class TransformerDecoder(Layer):
    def __init__(self):
        super().__init__()
        self.embed_dim = ModelSettings.EMBEDDING_DIMENSION.value
        self.latent_dim = ModelSettings.LATENT_DIMENSION.value
        self.num_heads = ModelSettings.NUMBER_HEADS.value
        self.attention_1 = MultiHeadAttention(
            num_heads=ModelSettings.NUMBER_HEADS.value, 
            key_dim=ModelSettings.EMBEDDING_DIMENSION.value
        )
        self.attention_2 = MultiHeadAttention(
            num_heads=ModelSettings.NUMBER_HEADS.value, 
            key_dim=ModelSettings.EMBEDDING_DIMENSION.value
        )
        self.dense_projtion = Sequential([
          Dense(ModelSettings.LATENT_DIMENSION.value, activation=ModelSettings.ACTIVATION.value), 
          Dense(ModelSettings.EMBEDDING_DIMENSION.value)
        ])
        self.layernorm_1 = LayerNormalization()
        self.layernorm_2 = LayerNormalization()
        self.layernorm_3 = LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = cast(mask[:, newaxis, :], dtype="int32")
            padding_mask = minimum(padding_mask, causal_mask)

        attended_inputs = self.attention_1(
            query=inputs, 
            value=inputs, 
            key=inputs, 
            attention_mask=causal_mask
        )
        decoder_inputs = self.layernorm_1(inputs + attended_inputs)

        attended_decoder_inputs = self.attention_2(
            query=decoder_inputs,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        outputs = self.layernorm_2(decoder_inputs + attended_decoder_inputs)

        projected_outputs = self.dense_projtion(outputs)
        return self.layernorm_3(outputs + projected_outputs)

    @staticmethod
    def get_causal_attention_mask(inputs):
        input_shape = shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = range(sequence_length)[:, newaxis]
        j = range(sequence_length)
        mask = cast(i >= j, dtype="int32")
        mask = reshape(mask, (1, sequence_length, sequence_length))
        multiple = concat([
            expand_dims(batch_size, -1), 
            constant([1, 1], dtype="int32")
        ],axis=0)
        return tile(mask, multiple)