from typing import List
from os.path import exists 

from tensorflow.keras.layers import Dense, Dropout
from tensorflow import range
from keras import Input, Model
from numpy import argmax, ndarray
from tensorflow.data import Dataset 

from conditional_generator_custom.model_layers.positional_embedding import PositionalEmbedding
from conditional_generator_custom.model_layers.transformer_decoder import TransformerDecoder
from conditional_generator_custom.model_layers.transformer_encoder import TransformerEncoder
from conditional_generator_custom.utils import ModelSettings, TOKENISER
from conditional_generator_custom.data_encoder import format_condition, input_vectorisation, output_vectorisation

class ConditionalGeneratorTransformer:
    def __init__(self) -> None:
        self.model = self._build_model()
        if exists(ModelSettings.MODEL_PATH.value): 
            self.model.load_weights(ModelSettings.MODEL_PATH.value)

    def train(self,data:Dataset, validation_data:Dataset) -> None:
        self.model.fit(data, epochs=ModelSettings.EPOCHS.value, validation_data=validation_data)

    def save(self) -> None:
        self.model.save_weights(ModelSettings.MODEL_PATH.value)

    def generate(self, style:str, sentiment:str, keywords:List[str]) -> str:
        input_condition = format_condition(style,sentiment,keywords)
        tokenised_input_condition = input_vectorisation([input_condition])
        decoded_sentence = ""

        for position in range(ModelSettings.MAXIMUM_GENERATION_LENGTH.value):
            tokenised_target_sentence = output_vectorisation([decoded_sentence])[:, :-1]

            logits = self.model([tokenised_input_condition, tokenised_target_sentence])

            predicted_token_id = self._greedy_decode(logits, position)
            predicted_token = str(TOKENISER.decode([predicted_token_id])) 
            decoded_sentence += f" {predicted_token}"

        return decoded_sentence
  
    @staticmethod
    def _greedy_decode(logits:ndarray,position:int) -> int:
        return argmax(logits[0, position, :])          

    @staticmethod
    def _build_model() -> Model:
        encoder_inputs = Input(shape=(None,), dtype="int64", name="encoder_inputs")
        contextualised_encoder_inputs = PositionalEmbedding()(encoder_inputs)
        encoder_outputs = TransformerEncoder()(contextualised_encoder_inputs)
        encoder = Model(encoder_inputs, encoder_outputs)

        decoder_inputs = Input(shape=(None,), dtype="int64", name="decoder_inputs")
        encoded_inputs = Input(shape=(None, ModelSettings.EMBEDDING_DIMENSION.value), name="decoder_state_inputs")
        contextualised_decoder_inputs = PositionalEmbedding()(decoder_inputs)
        projected_decoder_inputs = TransformerDecoder()(contextualised_decoder_inputs, encoded_inputs)
        decoder_logits = Dropout(ModelSettings.DROPOUT_RATE.value)(projected_decoder_inputs)
        decoder_outputs = Dense(2+len(TOKENISER), activation="softmax")(decoder_logits)
        decoder = Model([decoder_inputs, encoded_inputs], decoder_outputs)

        transformer_outputs = decoder([decoder_inputs, encoder_outputs])

        transformer = Model(
            [encoder_inputs, decoder_inputs], 
            transformer_outputs, 
            name="transformer"
        )
        transformer.compile(ModelSettings.OPTIMISER.value, loss=ModelSettings.LOSS.value, metrics=[ModelSettings.METRIC.value])
        return transformer