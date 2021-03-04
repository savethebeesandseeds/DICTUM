import re
import torch
import unidecode
from torch.nn import BCEWithLogitsLoss
from transformers import BertForSequenceClassification, BertTokenizer, tokenization_utils_base, AdamW

# LEarn to use the attention mask
## -----------
epochs = 50
max_length = 512
uncased_flag = False
to_train_flag = True
clean_text_data = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
def prepare_data(_data, _max_length):

    return 0
def get_input_data(_data):
    return tokenizer(
      [clean_text(_d[0]) if clean_text_data else _d[0] for _d in _data],
      max_length=max_length, # max of 512 in Bert
      truncation=tokenization_utils_base.TruncationStrategy("longest_first"), #['only_first', 'only_second', 'longest_first', 'do_not_truncate']
      add_special_tokens=True, # Add '[CLS]' and '[SEP]'
      return_token_type_ids=False,
      padding=tokenization_utils_base.PaddingStrategy("longest"), #['longest', 'max_length', 'do_not_pad']
      return_attention_mask=True,
      return_tensors='pt',  # Return PyTorch tensors = 'pt'
      verbose=True,
    ).to(torch.float)
def get_labels(_data):
    return torch.tensor([_d[1] for _d in _data]).to(torch.float)
## -----------
print(">>> Modelo para classificar secuencias de palabras.")
tokenizer = BertTokenizer.from_pretrained("BETO_UNCASED/" if uncased_flag else "BETO_CASED/", do_lower_case=False)
model = BertForSequenceClassification.from_pretrained("BETO_UNCASED/" if uncased_flag else "BETO_CASED/", return_dict=True, num_labels=3)
optimizer = AdamW(model.parameters(), lr=1e-2)
if(to_train_flag):
    for param in model.base_model.parameters():
        param.requires_grad = False
# print(model)
model.to(device)
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
data = [
    ("Para sanar los habitantes de Colombia, la población civil debe unirse de inmediato. Para todo lo demás existe libertad.",[0,0,0]),
    ("La rabia, el miedo, la lucha y la muerte.", [1,1,1])
]
# encoding = tokenizer.encode_plus(
encoding = get_input_data(data)
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']
## -----------
labels = get_labels(data)
print("input_size: {}, labels_size: {}".format(input_ids.size(),labels.size()))
input("STOP")
## -----------
if to_train_flag:
    model.train()
    for _e in range(epochs):
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = BCEWithLogitsLoss()(outputs.logits, labels)
        print("loss: {}".format(loss))
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
## -----------
model.eval()
outputs = model(input_ids, attention_mask=attention_mask)
loss = BCEWithLogitsLoss()(outputs.logits, labels)
logits = outputs.logits # Outputs
print("loss: {}".format(loss))
print("logits: {}".format(logits))
## -----------
