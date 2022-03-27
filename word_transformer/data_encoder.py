from typing import Tuple, List, Dict, Generator, Optional

from tensorflow import pad, Tensor, reshape

from word_transformer.utils import ModelSettings, TOKENISER
from dataset_creation.read.reader import Reader

def format_sentence(sentence:str) -> str:
  return sentence

def format_condition(style:str,sentiment:str,keywords:List[str]) -> str:
  return f"{style} {sentiment} {' '.join(keywords)}" 

def padded(values:List[int],max_length:int) -> Tensor:
    values = values[:max_length]
    pad_right = max_length-len(values)
    padded_values = pad(values,[(0,pad_right)],constant_values=ModelSettings.UNK_ID.value)
    return reshape(padded_values,shape=(1,-1))

input_vectorisation = lambda text: padded(values=TOKENISER.encode(str(text)).ids, max_length=ModelSettings.SEQUENCE_LENGTH.value) 
output_vectorisation =  lambda text: padded(values=TOKENISER.encode(str(text)).ids, max_length=1+ModelSettings.SEQUENCE_LENGTH.value) 

def get_training_pairs(size:Optional[int]=None) -> Generator[Tuple[Dict[str,str],str],None,None]:
    for input_condition,output_sentence in Reader(size).read(shuffled=False,skip_keywords_greater_than=ModelSettings.SEQUENCE_LENGTH.value-2):
        encoder_input_ids = input_vectorisation(format_condition(*input_condition))
        sentence_ids = output_vectorisation(format_sentence(output_sentence))
        decoder_input_ids = sentence_ids[:,:-1]
        decoder_output_ids = sentence_ids[:, 1:]
        inputs = dict(
            encoder_inputs= encoder_input_ids, 
            decoder_inputs= decoder_input_ids
        )
        yield inputs,decoder_output_ids