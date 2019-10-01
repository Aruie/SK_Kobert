from gluonnlp.data import SentencepieceTokenizer
from kobert.utils import get_tokenizer

def tokenizer(str, max_length = 512) :
    tok_path = get_tokenizer()
    sp  = SentencepieceTokenizer(tok_path)
    length = len(sp)

    return sp(str)

    