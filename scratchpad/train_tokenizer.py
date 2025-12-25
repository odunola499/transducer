import sentencepiece as spm
from datasets import load_dataset

dataset = load_dataset('wikimedia/wikipedia','20231101.en')['train']
texts = dataset['text']

with open('tokenizer_data.txt','w') as fp:
    for text in texts:
        fp.write(text)
        fp.write('\n')

spm.SentencePieceTrainer.train(
    input = 'tokenizer_data.txt',
    model_prefix = 'tokenizer_model',
    vocab_size = 1024,
)