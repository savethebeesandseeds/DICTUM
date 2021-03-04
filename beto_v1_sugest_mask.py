import re
import torch
import unidecode
from transformers import BertForMaskedLM, BertTokenizer, tokenization_utils_base
## -----------
max_length = 512
uncased_flag = False
## -----------
def clean_text(_text):
    if uncased_flag:
        # _text = unidecode.unidecode(_text)
        _text = _text.lower()
    _text = re.sub("\[mask\]", "[MASK]", _text)
    _text = re.sub("\[sep\]", "[SEP]", _text)
    _text = re.sub("\[cls\]", "[CLS]", _text)
    _text = re.sub("\[pad\]", "[PAD]", _text)
    _text = re.sub("\[unk\]", "[UNK]", _text)
    return _text
## -----------
print(">>> Modelo para completar espacios vacíos.")
tokenizer = BertTokenizer.from_pretrained("BETO_UNCASED/" if uncased_flag else "BETO_CASED/", do_lower_case=False)
model = BertForMaskedLM.from_pretrained("BETO_UNCASED/" if uncased_flag else "BETO_CASED/")
model.eval()
## -----------
# text = "[CLS] Para [MASK] los [MASK] de Colombia, la población civil debe [MASK] de inmediato. [SEP]"
# tokens = tokenizer.tokenize(text)
# masked_idxs = tuple([idx for idx,tk in enumerate(tokens) if tk.lower() == "[mask]"])
# indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
# tokens_tensor = torch.tensor([indexed_tokens])
# print(tokenizer.convert_ids_to_tokens(tokens_tensor[0]))
# print("TEXT: {}".format(text))
# predictions = model(tokens_tensor)[0]
# print(predictions)
# for i, midx in enumerate(masked_idxs):
#     idxs = torch.argsort(predictions[0,midx], descending=True)
#     predicted_token = tokenizer.convert_ids_to_tokens(idxs[:5])
#     print("MASK: {} : {}".format(i,predicted_token))
## -----------
print("-------------------")
text = "Para [MASK] los [MASK] de Colombia, la población civil debe [MASK] de inmediato. Para todo lo demás existe [MASK]."
text = clean_text(text)
print("TEXT(1): {}".format(text))
tokens_tensor = tokenizer.encode_plus(
  text,
  max_length=max_length,
  truncation=tokenization_utils_base.TruncationStrategy("longest_first"),
  add_special_tokens=True, # Add '[CLS]' and '[SEP]'
  return_token_type_ids=False,
  padding=True,
  return_attention_mask=True,
  return_tensors='pt',  # Return PyTorch tensors = 'pt'
  verbose=True,
)['input_ids']
masked_idxs = tuple([idx for idx,tk in enumerate(tokens_tensor[0]) if tk == 0]) # 0 == "[MASK]"
print(tokenizer.convert_ids_to_tokens(tokens_tensor[0]))
print("TEXT(2): {}".format(text))
predictions = model(tokens_tensor)[0]
for i, midx in enumerate(masked_idxs):
    idxs = torch.argsort(predictions[0,midx], descending=True)
    predicted_token = tokenizer.convert_ids_to_tokens(idxs[:5])
    print("MASK: {} : {}".format(i,predicted_token))
## -----------
