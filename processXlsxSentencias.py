## ---------
# Este codigo es propiedad de www.waajacu.com
# Hecho por Santiago Restrepo Ruiz.
# This is a Proprietary software; sharing, distributing 
# and selling are prohibit without explicit authorization 
# from waajacu.
## ---------
import math
import json
import pandas as pd
json_file_path = './SENTENCIAS.json'
sentencias_path = './SENTENCIAS.xlsx'
sentencias_pd_doc = pd.read_excel(sentencias_path,sheet_name='GENERAL')
# ----------------------
idx = 1
count = 1
newItem_flag = None
sentencia_obj = {}
all_sentencias = []
# ----------------------
for row in sentencias_pd_doc.iterrows():
    try:
        # ---------------
        if(not math.isnan(float(row[1]['Unnamed: 1']))):
            print("ELMNT: \t {} \t {}".format(type(row[1]['Unnamed: 1']),row[1]['Unnamed: 1']))
            count += 1
            if(newItem_flag is not None):
                all_sentencias.append(sentencia_obj)
            newItem_flag = True
            sentencia_obj = {
                'path':'',
                'excel_idx':[],
                'sentencia_idx':[],
                'sentencia_ano':[],
                'sentencia_estado':[],
                'sentencia_caso':[],
                'sentencia_hechos':[],
                'sentencia_ano2':[],
                'sentencia_contexto':[],
                'sentencia_subcontexto':[],
                'sentencia_palabrasClave':[],
                'sentencia_victimario':[],
                'sentencia_contextoVS':[],
                'sentencia_sentido':[],
                'sentencia_reconoceResponsabilidad':[],
                'sentencia_derechosVioladosCADHs':[],
                'sentencia_derechosNoViolados':[],
                'sentencia_violacionesDesestimadas':[],
                'sentencia_observaciones':[]
            }
        else:
            print("\t {} \t {} \t{}".format(type(row[1]['Unnamed: 1']),row[1]['Unnamed: 1'],count))
            if(newItem_flag is not None):
                newItem_flag = False
        # ---------------
        if(newItem_flag is not None):
            sentencia_obj['path'] = ''
            sentencia_obj['excel_idx'].append(                          str(row[0]))
            (sentencia_obj['sentencia_idx'].append(                     str(row[1]['Unnamed: 1']).strip().lower())  if  not math.isnan((float(1.0) if type(row[1]['Unnamed: 1'])  is not float else float(row[1]['Unnamed: 1'])))  else None)
            (sentencia_obj['sentencia_ano'].append(                     str(row[1]['Unnamed: 2']).strip().lower())  if  not math.isnan((float(1.0) if type(row[1]['Unnamed: 2'])  is not float else float(row[1]['Unnamed: 2'])))  else None)
            (sentencia_obj['sentencia_estado'].append(                  str(row[1]['Unnamed: 3']).strip().lower())  if  not math.isnan((float(1.0) if type(row[1]['Unnamed: 3'])  is not float else float(row[1]['Unnamed: 3'])))  else None)
            (sentencia_obj['sentencia_caso'].append(                    str(row[1]['Unnamed: 4']).strip().lower())  if  not math.isnan((float(1.0) if type(row[1]['Unnamed: 4'])  is not float else float(row[1]['Unnamed: 4'])))  else None)
            (sentencia_obj['sentencia_hechos'].append(                  str(row[1]['Unnamed: 5']).strip().lower())  if  not math.isnan((float(1.0) if type(row[1]['Unnamed: 5'])  is not float else float(row[1]['Unnamed: 5'])))  else None)
            (sentencia_obj['sentencia_ano2'].append(                    str(row[1]['Unnamed: 6']).strip().lower())  if  not math.isnan((float(1.0) if type(row[1]['Unnamed: 6'])  is not float else float(row[1]['Unnamed: 6'])))  else None)
            (sentencia_obj['sentencia_contexto'].append(                str(row[1]['Unnamed: 7']).strip().lower())  if  not math.isnan((float(1.0) if type(row[1]['Unnamed: 7'])  is not float else float(row[1]['Unnamed: 7'])))  else None)
            (sentencia_obj['sentencia_subcontexto'].append(             str(row[1]['Unnamed: 8']).strip().lower())  if  not math.isnan((float(1.0) if type(row[1]['Unnamed: 8'])  is not float else float(row[1]['Unnamed: 8'])))  else None)
            (sentencia_obj['sentencia_palabrasClave'].append(           str(row[1]['Unnamed: 8']).strip().lower())  if  not math.isnan((float(1.0) if type(row[1]['Unnamed: 8'])  is not float else float(row[1]['Unnamed: 8'])))  else None)
            (sentencia_obj['sentencia_victimario'].append(              str(row[1]['Unnamed: 10']).strip().lower()) if  not math.isnan((float(1.0) if type(row[1]['Unnamed: 10']) is not float else float(row[1]['Unnamed: 10']))) else None)
            (sentencia_obj['sentencia_contextoVS'].append(              str(row[1]['Unnamed: 11']).strip().lower()) if  not math.isnan((float(1.0) if type(row[1]['Unnamed: 11']) is not float else float(row[1]['Unnamed: 11']))) else None)
            (sentencia_obj['sentencia_sentido'].append(                 str(row[1]['Unnamed: 12']).strip().lower()) if  not math.isnan((float(1.0) if type(row[1]['Unnamed: 12']) is not float else float(row[1]['Unnamed: 12']))) else None)
            (sentencia_obj['sentencia_reconoceResponsabilidad'].append( str(row[1]['Unnamed: 13']).strip().lower()) if  not math.isnan((float(1.0) if type(row[1]['Unnamed: 13']) is not float else float(row[1]['Unnamed: 13']))) else None)
            (sentencia_obj['sentencia_derechosVioladosCADHs'].append(   str(row[1]['Unnamed: 14']).strip().lower()) if  not math.isnan((float(1.0) if type(row[1]['Unnamed: 14']) is not float else float(row[1]['Unnamed: 14']))) else None)
            (sentencia_obj['sentencia_derechosNoViolados'].append(      str(row[1]['Unnamed: 15']).strip().lower()) if  not math.isnan((float(1.0) if type(row[1]['Unnamed: 15']) is not float else float(row[1]['Unnamed: 15']))) else None)
            (sentencia_obj['sentencia_violacionesDesestimadas'].append( str(row[1]['Unnamed: 16']).strip().lower()) if  not math.isnan((float(1.0) if type(row[1]['Unnamed: 16']) is not float else float(row[1]['Unnamed: 16']))) else None)
            (sentencia_obj['sentencia_observaciones'].append(           str(row[1]['Unnamed: 17']).strip().lower()) if  not math.isnan((float(1.0) if type(row[1]['Unnamed: 17']) is not float else float(row[1]['Unnamed: 17']))) else None)
        # ---------------        
    except:
        # input("ERROR: \t {} \t {}".format(type(row[1]['Unnamed: 1']),row[1]['Unnamed: 1']))
        print("MEGAERROR:")
        raise
# --------------
with open(json_file_path, 'w', encoding='utf-8') as fp:
    json.dump(all_sentencias, fp, indent=4)
