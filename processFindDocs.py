## ---------
# Este codigo es propiedad de www.waajacu.com
# Hecho por Santiago Restrepo Ruiz.
# This is a Proprietary software; sharing, distributing 
# and selling are prohibit without explicit authorization 
# from waajacu.
## ---------
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import re
import json
import unidecode
from datetime import datetime
from PyPDF4 import PdfFileReader

# ---------
def assert_file(path, resetFlag=False):
    if(not(os.path.exists(path)) or resetFlag):
        with open(path, "w") as _:
            pass
    assert os.path.isfile(path)
def assert_folder(path, resetFlag=False):
    aux_str = "Error asserting folder: {}".format(path)
    if(resetFlag):
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(path)
        assert not(os.path.exists(path)), aux_str
    if(not(os.path.exists(path))):
        os.makedirs(path)
    assert os.path.exists(path), aux_str
# ---------
def get_info(pdfPath, byContent=True):
    filename = (os.path.splitext(os.path.basename(pdfPath))[0]).lower()
    page_content = ''
    if(byContent):
        pdf_document = PdfFileReader(open(pdfPath, 'rb'))
        for current_page in range(0, pdf_document.getNumPages()):
            page_content += pdf_document.getPage(current_page).extractText()
    # print(u'{}'.format(page_content.encode("utf8")))
    return filename, page_content

# ---------
def openJson(json_file_path):
    with open(json_file_path, 'r') as fp:
        data = json.load(fp)
    return data
def saveJson(json_data,json_file_path):
    with open(json_file_path, 'w', encoding='utf-8') as fp:
        json.dump(json_data, fp, indent=4)
# ---------
def identifyByFilename(filename, file, sentenciasData):
    for d_ in sentenciasData:
        idfiers = []
        idfiers.append([y for x in [unidecode.unidecode(dd_).lower().strip().split(" ") for dd_ in d_['sentencia_estado']] for y in x])
        idfiers.append([y for x in [unidecode.unidecode(dd_).lower().strip().split(" ") for dd_ in d_['sentencia_caso']] for y in x])
        idfiers = [y for x in idfiers for y in x]
        flags = [(s_ in unidecode.unidecode(filename).lower().strip()) for s_ in idfiers]
        # print(filename,idfiers,flags)
        # print("\t",flags)
        if(all(flags)):
            d_['path'].append(file)
            print("\t POSITIVE: (byFilename) {}".format(filename))
def identifyByFilecontent(filename, file, page_content, sentenciasData):
    for d_ in sentenciasData:
        idfiers = []
        idfiers.append([y for x in [unidecode.unidecode(dd_).lower().strip().split(" ") for dd_ in d_['sentencia_hechos']] for y in x])
        idfiers = [y for x in idfiers for y in x]
        factWords = unidecode.unidecode(page_content).lower().strip().split(" ")
        flags = [(s_ in factWords) for s_ in idfiers]
        # print(filename,idfiers,flags)
        # print("\t",flags)
        if(all(flags)):
            d_['path'].append(file)
            print("\t POSITIVE: (byFilecontent) {}".format(filename))
# ---------
if __name__ == "__main__":
    # ------
    byContent = False
    processFolder = r"./docs/dataBase/SENTENCIAS/"
    sentenciasFile = r"./SENTENCIAS.json"
    assert os.path.exists(sentenciasFile), "no JSON file found."
    sentenciasData = openJson(sentenciasFile)
    # ------- PROCESS ----
    for d_ in sentenciasData:
        d_['path'] = []
    for file_ in [os.path.join(dirpath,filename) for dirpath, _, filenames in os.walk(processFolder) for filename in filenames if filename.endswith('.pdf')]:
        print(">  Procesando archivo: {}".format(file_))
        filename_, page_content_ = get_info(file_, byContent)
        identifyByFilename(filename_, file_, sentenciasData)
        if(byContent):
            identifyByFilecontent(filename_, file_, page_content_, sentenciasData)
    # ------- SAVE ------
    print("------- NEGATIVE COMPROV -------")
    for d_ in sentenciasData:
        if(len(d_['path']) == 0):
            print("\t NEGATIVE: {} vs {}".format(d_['sentencia_caso'][0],d_['sentencia_estado'][0]))
    saveJson(sentenciasData,sentenciasFile)
