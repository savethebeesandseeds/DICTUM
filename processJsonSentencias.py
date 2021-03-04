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
import operator
import readline
import unidecode
from functools import reduce
from datetime import datetime
from PyPDF4 import PdfFileReader
import numpy as np
# ---------
def rlinput(prompt, prefill=''):
    readline.set_startup_hook(lambda: readline.insert_text(prefill))
    try:
        return input(prompt)  # or raw_input in Python 2
    except:
        assert False, "ERROR: Please use python3.6 or higher."
    finally:
        readline.set_startup_hook()
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
def openJson(json_file_path):
    with open(json_file_path, 'r') as fp:
        data = json.load(fp)
    return data
def saveJson(json_data,json_file_path):
    with open(json_file_path, 'w', encoding='utf-8') as fp:
        json.dump(json_data, fp, indent=4)
# ---------
def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))
def unique_array(a):
    aux = np.unique(a)
    if type(aux) is not list:
        aux = list(aux)
    return aux
def flat_listOfLists(a):
    return reduce(operator.concat, a)
# ---------
if __name__ == "__main__":
    # ------
    sentenciasFile = r"./SENTENCIAS.json"
    unique_fieldsFile = r"./uniqueFields.json"
    assert os.path.exists(sentenciasFile), "no JSON file found."
    sentenciasData = openJson(sentenciasFile)
    # ------
    print("------- COMPROV -------")
    need_save_flag = False
    total_cnt = 0
    negative_cnt = 0
    positive_cnt = 0
    not_sincere_cnt = 0
    _unique_fields = {
        'sentencia_contexto':[],
        'sentencia_subcontexto': [],
        'sentencia_palabrasClave': [],
        'sentencia_victimario': [],
        'sentencia_contextoVS': [],
        'sentencia_sentido': [],
        'sentencia_reconoceResponsabilidad': [],
        'sentencia_derechosVioladosCADHs': [],
        'sentencia_derechosNoViolados': [],
        'sentencia_violacionesDesestimadas': [],
        'sentencia_observaciones': []
    }
    # --------- PROCESS ------
    for d_ in sentenciasData:
        for d__ in [_ for _ in d_.keys() if _ != "path"]:
            if type(d_[d__]) is list:
                for d___ in range(len(d_[d__])):
                    d_[d__][d___] = unidecode.unidecode(d_[d__][d___]).lower().strip()
            else:
                d_[d__] = unidecode.unidecode(d_[d__]).lower().strip()
            need_save_flag = True
        d_["path"] = unique_array(d_["path"])
        if(len(d_['path'])>1):
            print("> (found)\tNOT-SINCERE: {} vs {}".format(d_['sentencia_caso'][0],d_['sentencia_estado'][0]))
            print("\t\tCASOS: <{} vs {}>:\n\t\t\t{}".format(d_['sentencia_caso'][0],d_['sentencia_estado'][0],json.dumps(d_['path'], indent=4, sort_keys=False)))
            aux_idx = 0
            for d__ in d_['path']:
                if(input("\t >>> select this <{}>: S/N (?): ".format(d__)).lower()=='s'):
                    d_['path'] = [d_['path'][aux_idx]]
                    need_save_flag = True
                    break
                aux_idx += 1
    # --------- SINCERE -------
    for d_ in sentenciasData:
        total_cnt += 1
        if(len(d_['path']) == 0):
            negative_cnt += 1
            print("\t-\t NEGATIVE: {} vs {}".format(d_['sentencia_caso'][0],d_['sentencia_estado'][0]))
        elif(len(d_['path']) == 1):
            positive_cnt += 1
            print("\t\t POSITIVE: {} vs {}".format(d_['sentencia_caso'][0],d_['sentencia_estado'][0]))
        else:
            not_sincere_cnt += 1
            print("\tx\t NOT-SINCERE: {} vs {}".format(d_['sentencia_caso'][0],d_['sentencia_estado'][0]))
            print("\t\tCASOS: <{} vs {}>:\n\t\t\t{}".format(d_['sentencia_caso'][0],d_['sentencia_estado'][0],json.dumps(d_['path'], indent=4, sort_keys=False)))
        # ----
    # --------- COMPUTE UNIQUE ------
    unique_fields = json.loads(json.dumps(_unique_fields))
    for d_ in sentenciasData:
        for q_ in unique_fields.keys():
            unique_fields[q_].append([s_ for s_ in d_[q_] if s_ not in unique_fields[q_]])
    for q_ in unique_fields.keys():
        unique_fields[q_] = flat_listOfLists(unique_fields[q_])
        unique_fields[q_] = unique_array(unique_fields[q_])
        for q__ in range(len(unique_fields[q_])):
            unique_fields[q_][q__] = unidecode.unidecode(unique_fields[q_][q__]).lower().strip()
    # --------- CORRECT LABELS -------
    if(input("\tCORECT LABELS S/N (?): ").lower()=='s'):
        print("\t...CORRECTING LABELS:\n")
        for q_ in unique_fields.keys():
            print(">> {}: \n\t {}".format(q_,json.dumps(unique_fields[q_], indent=4, sort_keys=False)))
            for q__ in range(len(unique_fields[q_])):
                inpt = rlinput("Type correct label for <{}>: ".format(q_),prefill=unique_fields[q_][q__])
                for d_ in [d_ for d_ in sentenciasData if inpt != unique_fields[q_][q__] and unique_fields[q_][q__] in d_[q_]]:
                    d_[q_] = [re.sub(unique_fields[q_][q__], inpt, k_) for k_ in d_[q_]]

    # --------- COMPUTE UNIQUE ------
    unique_fields = json.loads(json.dumps(_unique_fields))
    for d_ in sentenciasData:
        for q_ in unique_fields.keys():
            unique_fields[q_].append([s_ for s_ in d_[q_] if s_ not in unique_fields[q_]])
    for q_ in unique_fields.keys():
        unique_fields[q_] = flat_listOfLists(unique_fields[q_])
        unique_fields[q_] = unique_array(unique_fields[q_])
        for q__ in range(len(unique_fields[q_])):
            unique_fields[q_][q__] = unidecode.unidecode(unique_fields[q_][q__]).lower().strip()
    # --------- REPORT -------
    print("\t>> TOTAL: {}\tNEGATIVE: {}\tPOSITIVE: {}\tNOT_SINCERE: {}".format(total_cnt,negative_cnt,positive_cnt,not_sincere_cnt))
    for q_ in unique_fields.keys():
        print("\t >> {}: \n\t {}".format(q_,json.dumps(unique_fields[q_], indent=4, sort_keys=False)))
    # --------- SAVE ---------
    if(need_save_flag and input("\tSAVE S/N (?): ").lower()=='s'):
        saveJson(unique_fields, unique_fieldsFile)
        print("\t>> Sucessful save to file {}.".format(unique_fieldsFile))
        saveJson(sentenciasData,sentenciasFile)
        print("\t>> Sucessful save to file {}.".format(sentenciasFile))
