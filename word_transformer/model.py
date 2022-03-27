from typing import List
from os.path import exists 
from scipy.special import softmax 

from tensorflow.keras.layers import Dense, Dropout
from tensorflow import range
from keras import Input, Model
from tensorflow.data import Dataset 
from numpy import argmax, ndarray, argsort
from numpy.random import choice

from word_transformer.model_layers.positional_embedding import PositionalEmbedding
from word_transformer.model_layers.transformer_decoder import TransformerDecoder
from word_transformer.model_layers.transformer_encoder import TransformerEncoder
from word_transformer.utils import ModelSettings, TOKENISER
from word_transformer.data_encoder import format_condition, input_vectorisation, output_vectorisation

class ConditionalGeneratorTransformer:
    def __init__(self) -> None:
        self.model = self._build_model()
        if exists(ModelSettings.MODEL_PATH.value): 
            self.model.load_weights(ModelSettings.MODEL_PATH.value)

    def train(self,data:Dataset, validation_data:Dataset) -> None:
        for _ in range(ModelSettings.EPOCHS.value):
            self.model.fit(data, epochs=1, validation_data=validation_data)
            self.save()
            keywords = []
            print(self.generate("statement","positive",keywords))
            print(self.generate("statement","neutral",keywords))
            print(self.generate("statement","negative",keywords))
            print(self.generate("question","positive",keywords))
            print(self.generate("question","neutral",keywords))
            print(self.generate("question","negative",keywords))
            
    def save(self) -> None:
        self.model.save_weights(ModelSettings.MODEL_PATH.value)

    def generate(self, style:str, sentiment:str, keywords:List[str]) -> str:
        input_condition = format_condition(style,sentiment,keywords)
        tokenised_input_condition = input_vectorisation([input_condition])
        decoded_sentence = ""

        for position in range(ModelSettings.MAXIMUM_GENERATION_LENGTH.value):
            tokenised_target_sentence = output_vectorisation([decoded_sentence])[:, :-1]

            logits = self.model([tokenised_input_condition, tokenised_target_sentence])

            #predicted_token_id = self._greedy_decode(logits, position)
            predicted_token_id = self._top_k_decode(logits, position, 10)
            predicted_token = str(TOKENISER.decode([predicted_token_id])) 
            decoded_sentence += f" {predicted_token}"

        return decoded_sentence
  
    @staticmethod
    def _greedy_decode(logits:ndarray,position:int) -> int:
        return argmax(logits[0, position, :])     

    @staticmethod
    def _top_k_decode(logits:ndarray, position:int, top_k:int) -> int:
        values = logits[0,position,:]
        top_k_indexes = argsort(values)[-top_k:]
        probabilities = softmax(list(map(lambda index:values[index], top_k_indexes)))
        return choice(top_k_indexes,p=probabilities)

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