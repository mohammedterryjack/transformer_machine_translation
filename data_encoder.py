from typing import Tuple, List, Dict, Generator, Optional

from tensorflow.data import Dataset 
from tensorflow import pad, Tensor, reshape

from conditional_generator_custom.utils import ModelSettings, TOKENISER
from dataset_creation.read.reader import Reader

def format_sentence(sentence:str) -> str:
  return sentence

def format_condition(style:str,sentiment:str,keywords:List[str]) -> str:
  return f"{style} {sentiment} {' '.join(keywords)}" 

def get_training_pairs(size:Optional[int]) -> Generator[Tuple[str,str],None,None]:
    for input_condition,output_sentence in Reader(size).read(skip_keywords_greater_than=ModelSettings.SEQUENCE_LENGTH.value-2):
        yield format_condition(*input_condition),format_sentence(output_sentence)

def padded(values:List[int],max_length:int) -> Tensor:
    values = values[:max_length]
    pad_right = max_length-len(values)
    padded_values = pad(values,[(0,pad_right)],constant_values=UNK_ID)
    return reshape(padded_values,shape=(1,-1))

input_vectorisation = lambda text: padded(values=TOKENISER.encode(str(text)).ids, max_length=ModelSettings.SEQUENCE_LENGTH.value) #lambda text: TOKENISER.encode(str(text),return_tensors='tf',padding='max_length',max_length=ModelSettings.SEQUENCE_LENGTH.value, truncation=True)
output_vectorisation =  lambda text: padded(values=TOKENISER.encode(str(text)).ids, max_length=1+ModelSettings.SEQUENCE_LENGTH.value) 

def format_data(input_condition:str, output_sentence:str) -> Tuple[Dict[str,str],str]:
    encoder_input_ids = input_vectorisation(input_condition)
    sentence_ids = output_vectorisation(output_sentence)
    decoder_input_ids = sentence_ids[:,:-1]
    decoder_output_ids = sentence_ids[:, 1:]
    inputs = dict(
      encoder_inputs= encoder_input_ids, 
      decoder_inputs= decoder_input_ids
    )
    return inputs, decoder_output_ids

def make_dataset(size:Optional[int]=None) -> Dataset:
    pairs = get_training_pairs(size)
    train_inputs,train_outputs = zip(*pairs)
    dataset = Dataset.from_tensor_slices((list(train_inputs),list(train_outputs)))
    dataset = dataset.batch(ModelSettings.BATCH_SIZE.value)
    dataset = dataset.map(format_data)
    return dataset.shuffle(ModelSettings.SHUFFLE_SEED.value).prefetch(ModelSettings.PREFETCH.value).cache()