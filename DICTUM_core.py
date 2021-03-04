## ---------
# Este codigo es propiedad de www.waajacu.com
# Hecho por Santiago Restrepo Ruiz.
# This is a Proprietary software; sharing, distributing 
# and selling are prohibit without explicit authorization 
# from waajacu.
## ---------
# -*- coding: utf-8 -*-
import os
import re
from datetime import datetime
from PyPDF4 import PdfFileReader
# ----
import DICTUM_utils
# --------
def extract_info_YEAR(pdf_document):
    page_content = pdf_document.getPage(0).extractText()
    #RS = re.findall(r"(vs.|vs|Vs.|Vs|VS.|VS)(.*)",page_content)
    #input()
    #print(u'{}'.format(page_content.encode("utf8")))
    RS = re.findall(r"([0-9]{4})",page_content,flags=re.IGNORECASE | re.MULTILINE)
    RS = [x.strip() for x in RS if x != '']
    #print(RS)
    if(RS):
        print(">>\t\t<%s..............%s>" % ("AÑO:", RS[0]))
        return_var = RS[0]
    else:
        print(">>\tx\t<%s..............%s>" % ("AÑO:", "ERROR"))
        return_var = "Null"
    return return_var
def extract_info_ESTADO(pdf_document):
    page_content = pdf_document.getPage(0).extractText()
    #RS = re.findall(r"(vs.|vs|Vs.|Vs|VS.|VS)(.*)",page_content)
    #input()
    #print(u'{}'.format(page_content.encode("utf8")))
    RS = re.findall(r"vs\.?[(\n)(\n\S)(\n\S\n)(\.)]\.?(.*)$",page_content,flags=re.IGNORECASE | re.MULTILINE)
    RS = [x.strip() for x in RS if x != '']
    #print(RS)
    if(RS):
        print(">>\t\t<%s...........%s>" % ("ESTADO:", RS[0]))
        return_var = RS[0]
    else:
        print(">>\tx\t<%s...........%s>" % ("ESTADO:", "ERROR"))
        return_var = "Null"
    return return_var
def extract_info_CASO(pdf_document):
    page_content = pdf_document.getPage(0).extractText()
    page_content = DICTUM_utils.clean_text(page_content)
    #RS = re.findall(r"(vs.|vs|Vs.|Vs|VS.|VS)(.*)",page_content)
    #input()
    #print(u'{}'.format(page_content.encode("utf8")))
    RS = re.findall(r"caso(.*)vs.?",page_content,flags=re.IGNORECASE | re.MULTILINE)
    #print(RS)
    RS = [x.strip() for x in RS if x != '']
    if(RS):
        print(">>\t\t<%s.............%s>" % ("CASO:", RS[0]))
        return_var = RS[0]
    else:
        print(">>\tx\t<%s.............%s>" % ("CASO:", "ERROR"))
        return_var = "Null"
    return return_var
def extract_info_PALABRAS_CLAVES(pdf_document,palabras_clave):
    palabras_clave = [k.lower() for k in palabras_clave]
    keys = []
    for current_page in range(0, pdf_document.getNumPages()):
        page_content = pdf_document.getPage(current_page).extractText()
        page_content = DICTUM_utils.clean_text(page_content)
        RS = re.findall("("+")|(".join(palabras_clave)+")",page_content.lower(),flags=re.IGNORECASE)
        keys.append([list(r) for r in RS])
    keys = [item for sublist in keys for item in sublist]
    keys = [item for sublist in keys for item in sublist if item is not '']
    keys = list(set(keys))
    if(keys):
        print(">>\t\t<%s...%s" % ("PALABRAS CLAVE:", "["))
        for k in keys:
            print(">>\t\t                    %s" % (k))
        print(">>\t\t                    %s>" % ("]"))
    else:
        print(">>\tx\t<%s...%s>" % ("PALABRAS CLAVE:", ["-"]))
    return keys
def extract_info_GREEK(pdf_document,greek_num):
    greek_num_ = [k.lower() for k in greek_num]
    keys = []
    for current_page in range(0, pdf_document.getNumPages()):
        page_content = pdf_document.getPage(current_page).extractText()
        page_content = DICTUM_utils.clean_text(page_content)
        RS = re.findall(r"[\n-]("+")|(".join(greek_num_)+")",page_content.lower(),flags=re.IGNORECASE)
        keys.append([list(r) for r in RS])
    keys = [item for sublist in keys for item in sublist]
    keys = [item for sublist in keys for item in sublist if item is not '']
    keys = list(set(keys))
    if(keys):
        print(">>\t\t<%s...%s" % ("greek_num:", "["))
        for k in keys:
            print(">>\t\t                    %s" % (k))
        print(">>\t\t                    %s>" % ("]"))
    else:
        print(">>\tx\t<%s...%s>" % ("greek_num:", ["-"]))
    return keys
# --------
def extract_info_all(pdfPath,palabras_clave=None,greek_num=None):
    aux_str = "Error processing file (file not found): {}".format(pdfPath)
    assert os.path.isfile(pdfPath), aux_str
    aux_str = "Error processing file (file is not pdf): {}".format(pdfPath)
    assert pdfPath.endswith('.pdf'), aux_str
    pdf_document = PdfFileReader(open(pdfPath, 'rb'))
    # ----
    return_dict = {}
    return_dict["estado"] = extract_info_ESTADO(pdf_document)
    return_dict["año"] = extract_info_YEAR(pdf_document)
    return_dict["caso"] = extract_info_CASO(pdf_document)
    return_dict["palabras clave"] = extract_info_PALABRAS_CLAVES(pdf_document,palabras_clave)
    return_dict["greek"] = extract_info_GREEK(pdf_document,greek_num)
    print(">> core_STATUS_{}".format("OK"))
    # ----
    return return_dict
