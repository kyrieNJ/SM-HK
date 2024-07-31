import jieba
import numpy as np
from transformers import BertTokenizer
from vocab import Vocab
import spacy
nlp = spacy.load('zh_core_web_sm')
tokenizer = BertTokenizer.from_pretrained("./bert-chinese")
# tokenizer = BertTokenizer.from_pretrained("./mlm/bert-de_finetune")

def text_to_bert_sequence(text, max_len, padding="post", truncating="post"):
    text = tokenizer.tokenize(text)
    text = ["[CLS]"] + text + ["[SEP]"]
    sequence = tokenizer.convert_tokens_to_ids(text)
    return pad_and_truncate(sequence, max_len, padding=padding, truncating=truncating)


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc

    return x



def Get_bertout(text,vocab_dir):
    token_vocab = Vocab.load_vocab(vocab_dir + "_vocab_tok.vocab")
    text_indices = [token_vocab.stoi.get(t.text, token_vocab.unk_index) for t in nlp(text)]

    text_len = np.count_nonzero(text_indices)+10
    if text_len>512:
        text_len=512
    bert_sequence= text_to_bert_sequence(text, text_len)

    return bert_sequence