## ---------
# Este codigo es propiedad de www.waajacu.com
# Hecho por Santiago Restrepo Ruiz.
# This is a Proprietary software; sharing, distributing 
# and selling are prohibit without explicit authorization 
# from waajacu.
## ---------
import os
import re
import json
import torch
import random
import pickle
import unidecode
from PyPDF4 import PdfFileReader
from torch.nn import BCEWithLogitsLoss
from transformers import BertForSequenceClassification, BertTokenizer, tokenization_utils_base, AdamW

# fix labels to vectors, construct better a network with multiplexed output channels
# assert output shape conditions for when resume training
# better fix save and load model to _modelFile
## -------------
num_epochs          = 4
batch_size          = 100
max_length          = 512
learningRate        = 1e-4
validationSplit     = 0.1
uncased_flag        = False
to_train_flag       = True
clean_text_data     = True
## -------------
sentenciasFile      = r"./tempData/SENTENCIAS.json"
unique_fieldsFile   = r"./tempData/uniqueFields.json"
modelFile           = r"model.pkl"
## -------------
def clean_text_content(_text):
    if uncased_flag:
        # _text = unidecode.unidecode(_text)
        _text = _text.lower()
    _text = re.sub("\\n", "", _text)
    _text = re.sub("\\r", "", _text)
    _text = re.sub("\ \ \ ", " ", _text)
    _text = re.sub("\ \ ", " ", _text)
    return _text
def clean_text_masks(_text):
    _text = re.sub("\[mask\]", "[MASK]", _text)
    _text = re.sub("\[sep\]", "[SEP]", _text)
    _text = re.sub("\[cls\]", "[CLS]", _text)
    _text = re.sub("\[pad\]", "[PAD]", _text)
    _text = re.sub("\[unk\]", "[UNK]", _text)
    return _text
def get_doc_data(pdfPath, pages="all", print_clear=False):
    doc_content = ''
    pdf_document = PdfFileReader(open(pdfPath, 'rb'))
    for current_page in range(0, pdf_document.getNumPages()) if pages.lower()=="all" else pages:
        doc_content += pdf_document.getPage(current_page).extractText()
    if(print_clear):
        print(u'DOCUMENT <{}> ::: \n\t{}'.format((os.path.splitext(os.path.basename(pdfPath))[0]).lower(),doc_content.encode("utf8")))
    return doc_content
## -------------
def openJson(json_file_path):
    with open(json_file_path, 'r') as fp:
        data = json.load(fp)
    return data
def saveJson(json_data,json_file_path):
    with open(json_file_path, 'w', encoding='utf-8') as fp:
        json.dump(json_data, fp, indent=4)
## -------------
def saveVariable(_variable, _filePath):
    with open(_filePath, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(_variable, f)
def loadVariable(_filePath):
    with open(_filePath, 'rb') as f:  # Python 3: open(..., 'rb')
        _aux = pickle.load(f)
    return _aux
## -------------
class DICTUM_NETWORK(object):
    """docstring for DICTUM_NETWORK."""
    def __init__(self, _modelPath, _unique_fields, _uncased_flag=False, _clean_text_data=False, _max_length=512, _learningRate=1e-5):
        ## ----
        # _modelPath: path to model
        # _unique_fields: {'clsses':['list of possible values']}
        # _uncased_flag: wheater to use Bert (or BETO) cased or uncased
        # _clean_text_data: wheater or not to apply cleanning of text data
        # _max_length: max lenght of sentences in Bert (or BETO) model
        # _learningRate: learning rate for training the network
        ## ----
        self._modelPath = _modelPath
        self._unique_fields = _unique_fields
        self._uncased_flag = _uncased_flag
        self._clean_text_data = _clean_text_data
        self._max_length = _max_length
        self._learningRate = _learningRate
        ## ----
        self._training_performance = []
        self._validation_performance = []
        self._num_labels = len([True for q_ in self._unique_fields.keys() for q__ in self._unique_fields[q_]])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = BertTokenizer.from_pretrained("BETO_UNCASED/" if self._uncased_flag else "BETO_CASED/", do_lower_case=False)
        self.model = BertForSequenceClassification.from_pretrained("BETO_UNCASED/" if uncased_flag else "BETO_CASED/", return_dict=True, num_labels=self._num_labels)
        self.optimizer = AdamW(self.model.parameters(), lr=self._learningRate)
        self.model.to(self.device)
        ## ----
        for param in self.model.base_model.parameters():
            param.requires_grad = False
        # print(model)
    # --------
    def predict(self, _evalDataFile, printResult_flag=False):
        self._data = self.prepare_data(openJson(_evalDataFile), _shuffle=False)
        self.input_data = self.make_network_input_data()
        del self._data
        self.model.eval()
        outputs = model(self.input_data["input_ids"], attention_mask=self.input_data["attention_mask"]).to(torch.float)
        if printResult_flag:
            print("logits (output): {}".format(outputs.logits))
        del self.input_data
        return outputs
    def train(self, _trainDataFile=None, _validationSplit=0.0, _batch_size=100, _num_epochs= 5000, _trainFlag=True, _validationFlag=True, _saveFlag=True, _trainOnValidationData=False):
        ## ----
        ## -------- TRAINING
        if(_trainDataFile is None):
            print("No training data given.")
        else:
            self.model.train()
            if(_trainFlag):
                print("Starting training on {}/1.0 of file {}".format(_validationSplit,_trainDataFile))
                self._data = self.prepare_data(openJson(_trainDataFile), _portion=(0.0, 1.0-_validationSplit),_shuffle=True)
                self.training_input_data = self.make_network_input_data()
                self.training_labels_data = self.make_network_labels_data()
                del self._data
                print("trainning data conditions: num_epochs: {}, input_size: {}, labels_size: {}".format(_num_epochs,self.training_input_data["input_ids"].size(),self.training_labels_data.size()))
                assert len(self.training_input_data["input_ids"]) == len(self.training_labels_data), "Not matching data dimensions to labels for training."
                ## ----
                if(_trainFlag or input("\INITILIZE TRAINING S/N (?): ").lower()=='s'):
                    for _e in range(_num_epochs):
                        for _b in range(0, len(self.training_input_data["input_ids"]), _batch_size):
                            outputs = self.model(self.training_input_data["attention_mask"][_b:_b+_batch_size], attention_mask=self.training_input_data["attention_mask"][_b:_b+_batch_size])
                            self.loss = BCEWithLogitsLoss()(outputs.logits.to(torch.float), self.training_labels_data[_b:_b+_batch_size])
                            self._training_performance.append(self.loss)
                            print("epoch: {}/{}, batch: {}/{}, loss: {}".format(int(_e+1), _num_epochs, int(_b/_batch_size+1), int(len(self.training_input_data["input_ids"])/_batch_size)+1, self.loss))
                            self.loss.backward(retain_graph=True)
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                        if(_saveFlag):
                            saveVariable(self, self._modelPath)
                            print("Model saved for epoch {}..".format(_e+1))
                else:
                    print("Training aborted.")
                ## --------
                del self.training_input_data
                del self.training_labels_data
                del self.loss
                print("Training ended...")
            ## -------- VALIDATION
            if (_validationSplit <= 0.0):
                print("No validation data given.")
            elif(_validationFlag or _trainOnValidationData):
                print("Starting validation on {}/1.0 of file {}".format(1-_validationSplit,_trainDataFile))
                self._validation_performance = []
                self._data = self.prepare_data(openJson(_trainDataFile), _portion=(1-_validationSplit, 1.0), _shuffle=False)
                self.validation_input_data = self.make_network_input_data()
                self.validation_labels_data = self.make_network_labels_data()
                del self._data
                print("validation data conditions: input_size: {}, labels_size: {}".format(self.validation_input_data["input_ids"].size(),self.validation_labels_data.size()))
                assert len(self.validation_input_data["input_ids"]) == len(self.validation_labels_data), "Not matching data dimensions to labels for validation."
                self.model.eval()
                ## --------
                if(_validationFlag):
                    with torch.no_grad():
                        for _b in range(0, len(self.validation_input_data["input_ids"]), _batch_size):
                            outputs = self.model(self.validation_input_data["attention_mask"][_b:_b+_batch_size], attention_mask=self.validation_input_data["attention_mask"][_b:_b+_batch_size])
                            self.loss = BCEWithLogitsLoss()(outputs.logits.to(torch.float), self.validation_labels_data[_b:_b+_batch_size])
                            self._validation_performance.append(self.loss)
                            ## --------
                            print("Current performance for validation data : <{}> in batch {}/{} is: {}".format(_trainDataFile, int(_b/_batch_size+1), int(len(self.validation_input_data["input_ids"])/_batch_size)+1, self.loss))
                    print("Validations ended...")
                ## --------
                if(_trainOnValidationData):
                    print("Training on validation data...")
                    self.model.train()
                    print("trainning conditions: num_epochs: {}, input_size: {}, labels_size: {}".format(_num_epochs,self.validation_input_data["input_ids"].size(),self.validation_labels_data.size()))
                    assert len(self.validation_input_data["input_ids"]) == len(self.validation_labels_data), "Not matching data dimensions to labels for training."
                    ## ----
                    for _e in range(_num_epochs):
                        for _b in range(0, len(self.validation_input_data["input_ids"]), _batch_size):
                            outputs = self.model(self.validation_input_data["attention_mask"][_b:_b+_batch_size], attention_mask=self.validation_input_data["attention_mask"][_b:_b+_batch_size])
                            self.loss = BCEWithLogitsLoss()(outputs.logits.to(torch.float), self.validation_labels_data[_b:_b+_batch_size])
                            self._training_performance.append(self.loss)
                            print("epoch: {}/{}, batch: {}/{}, loss: {}".format(int(_e+1), _num_epochs, int(_b/_batch_size+1), int(len(self.validation_input_data["input_ids"])/_batch_size)+1, self.loss))
                            self.loss.backward(retain_graph=True)
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                        if(_saveFlag):
                            saveVariable(self, self._modelPath)
                            print("Model saved for epoch {}..".format(_e+1))
                    ## --------
                    print("Training on validation data ended...")
                del self.validation_input_data
                del self.validation_labels_data
                del self.loss
                if(_saveFlag):
                    saveVariable(self, self._modelPath)
                    print("Model saved after validation..")
            # ----
    # --------
    def prepare_data(self,_sentenciasData, _portion=None, _shuffle=False):
        # _sentenciasData: in the form of SENTENCIAS.json format
        # _portion: tiple (ini, strat) in data[ini:stop]
        # _shuffle: Flag, to shuffle data.
        print("Preparing data...")
        prepared_data = []
        for d_ in [p_ for p_ in _sentenciasData if len(p_["path"])==1] if (_portion == None) else [p_ for p_ in _sentenciasData[int(_portion[0]*len(_sentenciasData)):int(_portion[1]*len(_sentenciasData))] if len(p_["path"])==1]:
            item_data = [[],{}]
            for q_ in self._unique_fields.keys():
                item_data[1][q_] = dict((q__, 1) if q__ in d_[q_] else (q__, 0) for q__ in self._unique_fields[q_])
            _aux = get_doc_data(d_["path"][0], pages="all")
            if(self._clean_text_data):
                _aux = clean_text_content(_aux)
            item_data[0] = [_aux[k_:k_+self._max_length] for k_ in range(0, len(_aux), self._max_length)]
            prepared_data.append(item_data)
        if(_shuffle):
            prepare_datad = random.shuffle(prepared_data)
        return prepared_data
    def vector_to_labels(self, item, results_flag=True):
        return {q_:{q__:next(item) for q__ in self._unique_fields[q_]} for q_ in self._unique_fields.keys()}
    def labels_to_vector(self, item, results_flag=False):
        return [item[q_][q__] for q_ in self._unique_fields.keys() for q__ in self._unique_fields[q_]]
    # --------
    def make_network_labels_data(self):
        return torch.tensor([self.labels_to_vector(_d[1],results_flag=False) for _d in self._data for _ in range(0,len(_d[0]))]).to(torch.float)
    def make_network_input_data(self):
        # encoding = tokenizer.encode_plus(
        return self.tokenizer(
          [clean_text_masks(__d) for _d in self._data for __d in _d[0]],
          max_length=self._max_length, # max of 512 in Bert
          truncation=tokenization_utils_base.TruncationStrategy("longest_first"), #['only_first', 'only_second', 'longest_first', 'do_not_truncate']
          add_special_tokens=True, # Add '[CLS]' and '[SEP]'
          return_token_type_ids=False,
          padding=tokenization_utils_base.PaddingStrategy("longest"), #['longest', 'max_length', 'do_not_pad']
          return_attention_mask=True,
          return_tensors='pt',  # Return PyTorch tensors = 'pt'
          verbose=False,
        )
## -------------
if __name__ == "__main__":
    assert os.path.exists(sentenciasFile), "no JSON file found for Sentencias."
    assert os.path.exists(unique_fieldsFile), "no JSON file found for unique_fields"
    ## -------------
    print(">>> Modelo para classificar secuencias de palabras.")
    if(os.path.exists(modelFile)):
        dictum = loadVariable(modelFile)
        print("Loaded dictum model from file {}".format(modelFile))
    else:
        dictum = DICTUM_NETWORK(_modelPath=modelFile, _unique_fields=openJson(unique_fieldsFile), _uncased_flag=uncased_flag, _clean_text_data=clean_text_data,_max_length=max_length, _learningRate=learningRate)
        print("New dictum model created..")
    ## -------------
    if(to_train_flag):
        dictum.train(_trainDataFile=sentenciasFile, _validationSplit=validationSplit, _batch_size=batch_size, _num_epochs=num_epochs, _trainFlag=True, _validationFlag=True, _saveFlag=True, _trainOnValidationData=False)
    ## -------------
    ## -------------
