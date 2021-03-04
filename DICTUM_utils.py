## ---------
# Este código es propiedad de www.waajacu.com
# Hecho por Santiago Restrepo Ruiz.
# This is a Proprietary software; sharing, distributing 
# and selling are prohibit without explicit authorization 
# from waajacu.
## ---------
# -*- coding: utf-8 -*-
import os
import re
import json
import pickle
from datetime import datetime
from PyPDF4 import PdfFileReader


## -------------
def assert_file(_path, resetFlag=False):
    if(not(os.path.exists(_path)) or resetFlag):
        with open(_path, "w") as _:
            pass
    assert os.path.isfile(_path)
def assert_folder(_path, resetFlag=False):
    aux_str = "Error asserting folder: {}".format(_path)
    if(resetFlag):
        for root, dirs, files in os.walk(_path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(_path)
        assert not(os.path.exists(_path)), aux_str
    if(not(os.path.exists(_path))):
        os.makedirs(_path)
    assert os.path.exists(_path), aux_str
## -------------
def clean_text(_text):
    # _text = _text.encode("utf8")
    _text = _text.replace('\n','')
    _text = _text.replace('\r','')
    _text = _text.replace('  ',' ')
    _text = _text.replace('   ',' ')
    _text = _text.replace('    ',' ')
    _text = _text.replace('     ',' ')
    _text = _text.replace('      ',' ')
    _text = _text.replace('       ',' ')
    _text = _text.replace('        ',' ')
    # _text = _text.encode("utf8")
    return _text
## -------------
def clean_text_content(_text, _uncased_flag):
    if(_uncased_flag):
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
## -------------
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
    with open(_filePath, 'wb') as f:
        pickle.dump(_variable, f)
def loadVariable(_filePath):
    with open(_filePath, 'rb') as f:
        _aux = pickle.load(f)
    return _aux
## -------------
def print_page(pdfPath, page=None):
    pdf_document = PdfFileReader(open(pdfPath, 'rb'))
    page_content = pdf_document.getPage(page).extractText()
    print(u'{}'.format(page_content.encode("utf8")))
def find_text(search_term,pdfPath=None,pdf_document=None):
    assert not((pdfPath is None) and (pdf_document is None)), "ERROR, give a pdf_path or a pdf_document to function find_text() in main."
    if(not(pdf_document)):
        pdf_document = PdfFileReader(open(pdfPath, 'rb'))
    for current_page in range(0, pdf_document.getNumPages()):
        page_content = pdf_document.getPage(current_page).extractText()
        RS = re.search(search_term.lower(), page_content.lower())
        if(RS):
            print(">> <%s> found on page <%i>" % (search_term, current_page))
