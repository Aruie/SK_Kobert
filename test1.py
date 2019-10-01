import torch
from kobert.pytorch_kobert import get_pytorch_kobert_model



from tokenizer_test import tokenizer



input_ids = torch.LongTensor([[31,51,99],[15,5,0]])
input_mask = torch.LongTensor([[1,1,1],[1,1,0]])
token_type_ids = torch.LongTensor([[0,0,1],[0,1,0]])
model, vocab  = get_pytorch_kobert_model()
all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)

print(all_encoder_layers[-1][0])


input1 = vocab(tokenizer('중학교를 졸업하면 어디로 갑니까'))
len(input1)
print(input1)
mask1 = [1,1,1,1,1,1,1,1,1,1]
input2 = vocab(tokenizer('아마 다음엔 고등학교에 갑니다'))
print(input2)
len(input2)
mask2 = [1,1,1,1,1,1,1,1,1,0]


_, out = model(torch.LongTensor([input1,input2]), torch.LongTensor([mask1,mask2]))

