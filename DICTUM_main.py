## ---------
# Este codigo es propiedad de www.waajacu.com
# Hecho por Santiago Restrepo Ruiz.
# This is a Proprietary software; sharing, distributing 
# and selling are prohibit without explicit authorization 
# from waajacu.
## ---------
# -*- coding: utf-8 -*-
from os import path, chdir, listdir, remove
import argparse
import configparser
from datetime import datetime
# --------
import DICTUM_core
import DICTUM_utils
import DICTUM_network
from DICTUM_network import DICTUM_NETWORK
# --------
# Configure Initial Variables
config_file = 'nconfig.cfg'
# Configure Enviroment
config_file = path.join(path.abspath(path.dirname(__file__)), config_file)
current_dir_path = path.abspath(path.dirname(__file__))
if (current_dir_path != path.abspath(path.dirname(__file__))):
    chdir(current_dir_path)
assert current_dir_path == path.abspath(path.dirname(__file__))
assert path.isfile(config_file)
config = configparser.ConfigParser()
config.read(path.join(path.abspath(path.dirname(__file__)), config_file))
# --------
# ---------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "--v", default= False, action="store_true", help="Increase output verbosity")
    parser.add_argument("--toTrianNetwork", "--train",  type=lambda s: s.lower().strip() in ['true', 'si', 'yes', '1'], default= config.get("DICTUM","TO_TRAIN_NETWORK"), help="Flag to train network on sentencias file path.")
    parser.add_argument("--toCleanTextData", "--ctd",   type=lambda s: s.lower().strip() in ['true', 'si', 'yes', '1'], default= config.get("DICTUM","TO_CLEAN_TEXT_DATA"), help="Flag to clean text data for network.")
    parser.add_argument("--useUncasedModel", "--uum",   type=lambda s: s.lower().strip() in ['true', 'si', 'yes', '1'], default= config.get("DICTUM","USE_UNCASED_MODEL"), help="Flag to select uncased version of network.")
    parser.add_argument("--trainOnValidationData", "--tonvd", type=lambda s: s.lower().strip() in ['true', 'si', 'yes', '1'], default= config.get("DICTUM","TRAIN_ON_VALIDATION_DATA"), help="Flag to train on validation data.")
    parser.add_argument("--doValidationFlag", "--dvf",  type=lambda s: s.lower().strip() in ['true', 'si', 'yes', '1'], default= config.get("DICTUM","DO_VALIDATION_FLAG"), help="Flag to switch the validation procedure on the training function.")
    parser.add_argument("--doTrainFlag", "--dtf",       type=lambda s: s.lower().strip() in ['true', 'si', 'yes', '1'], default= config.get("DICTUM","DO_TRAINING_FLAG"), help="Flag to switch the actual training procedure on the training function.")
    parser.add_argument("--saveTraningFlag", "--stf",   type=lambda s: s.lower().strip() in ['true', 'si', 'yes', '1'], default= config.get("DICTUM","SAVE_TRAINED_MODEL"), help="Flag to switch the saveing of the model to the model file.")

    # parser.add_argument("--principalFile", "--file",    type=str, help="Path to input '.pdf' file.")
    parser.add_argument("--outputFile","--of",          type=str, default=config.get("DICTUM","FINAL_OUTPUT_FILE"),             help="Path to output json file.")
    parser.add_argument("--logfolder", "--lf",          type=str, default=config.get("DICTUM","LOG_FOLDER"),                    help="Path to folder where to store log files.")
    parser.add_argument("--docsfolder", "--df",         type=str, default=config.get("DICTUM","DOCS_FOLDER"),                   help="Path to folder where to store log files.")
    parser.add_argument("--modelPath", "--mp",          type=str, default=config.get("DICTUM","MODEL_PATH"),                    help="Path to dictum network model.")
    parser.add_argument("--sentenciasFilePath", "--sfp",type=str, default=config.get("DICTUM","SENTENCIAS_FILE_PATH"),          help="Path to json file of processed sentencias to support network.")
    parser.add_argument("--unique_fieldsFilePath", "--uffp",type=str, default=config.get("DICTUM","UNIQUE_FIELDS_FILE_PATH"),   help="Path to json file of unique fields to support network.")
    
    parser.add_argument("--num_epochs",     type=int,   default=config.get("DICTUM","TRAIN_BY_NUM_EPOCHS"),                     help="Numbers of epochs while training network.")
    parser.add_argument("--learningRate",   type=float, default=config.get("DICTUM","TRAIN_BY_LEARNING_RATE"),                  help="Learning rate while training network.")
    parser.add_argument("--validationSplit",type=float, default=config.get("DICTUM","TRAIN_BY_VALIDATION_SPLIT"),               help="Separation split, [0, 1], float. Size for training, 1-val is for training.")
    parser.add_argument("--batch_size",     type=int,   default=config.get("DICTUM","NETWORK_BY_BATCH_SIZE"),                   help="Batch size for processing while training or processing.")
    parser.add_argument("--max_length",     type=int,   default=config.get("DICTUM","NETWORK_SEQUENCE_INPUT_MAX_LENGTH"),       help="Input max size for network imput, a max of 512 for BERT.")

    args = vars(parser.parse_args())
    # ----
    if args["verbose"]:
        print("__WORKING DIRECTORY__: {}".format(current_dir_path))
        print("__CONFIG FILE__: {}".format(config_file))
        print("__CONFIG FILE DATA__:")
        for i in ["\t{}: \n\t\t{}".format(section, dict(config.items(section))) for section in dict(config.items()).keys()]:
            print(i)
        print("__SCRIPT ARGUMENTS__: {}".format(args))
    # ----
    final_retun_path    = args["outputFile"]
    logFolder           = args["logfolder"]
    docsFolder          = args["docsfolder"]
    errorFolder         = path.join(docsFolder,"error")
    processFolder       = path.join(docsFolder,"procesar")
    processedFolder     = path.join(docsFolder,"procesado")
    date_isoFormat      = datetime.utcnow().replace(tzinfo=None, microsecond=0).isoformat()
    logFile_Path        = path.join(logFolder, "logFile__{}.txt".format(date_isoFormat))
    # ---- 
    palabras_clave      = [param.strip() for param in config.get("DICTUM","PALABRAS_CLAVE").split(',')]
    puntos_resolutivos  = ['PUNTOS RESOLUTIVOS']
    greek_num           = [param.strip() for param in config.get("DICTUM","GREEK_NUMS").split(',')]
    # ---- Network Files ----
    modelFile           = args["modelPath"]
    sentenciasFile      = args["sentenciasFilePath"]
    unique_fieldsFile   = args["unique_fieldsFilePath"]
    # ---- Network Variables ----
    train_network       = args["toTrianNetwork"]
    clean_text_data     = args["toCleanTextData"]
    uncased_flag        = args["useUncasedModel"]
    num_epochs          = args["num_epochs"]
    batch_size          = args["batch_size"]
    max_length          = args["max_length"]
    learningRate        = args["learningRate"]
    validationSplit     = args["validationSplit"]
    trainOnValidationData = args["trainOnValidationData"]
    doValidationFlag    = args["doValidationFlag"]
    doTrainFlag         = args["doTrainFlag"]
    saveTraningFlag     = args["saveTraningFlag"]
    # ----
    assert max_length      >  0 and max_length      <= 512,  "Configure Max length in the range [1,512]."
    assert validationSplit >= 0 and validationSplit <= 1,    "Configure Validation Split in the range [0,1]."
    # ----
    DICTUM_utils.assert_folder(logFolder,False)
    DICTUM_utils.assert_folder(docsFolder,False)
    DICTUM_utils.assert_folder(errorFolder,False)
    DICTUM_utils.assert_folder(processFolder,False)
    DICTUM_utils.assert_folder(processedFolder,False)
    # ----
    DICTUM_utils.assert_file(final_retun_path, True)
    DICTUM_utils.assert_file(logFile_Path, True)
    # ----
    assert path.exists(sentenciasFile), "no JSON file found for Sentencias."
    assert path.exists(unique_fieldsFile), "no JSON file found for unique_fields"
    # ----
    if(path.exists(modelFile)):
        dictum_model = DICTUM_utils.loadVariable(modelFile)
        print("Loaded dictum model from file {}".format(modelFile))
    else:
        dictum_model = DICTUM_network.DICTUM_NETWORK(_modelPath=modelFile, _unique_fields=DICTUM_utils.openJson(unique_fieldsFile), _uncased_flag=uncased_flag, _clean_text_data=clean_text_data,_max_length=max_length, _learningRate=learningRate)
        print("New dictum model created..")
    # ---- ENTRENAR ----
    if(train_network):
        dictum_model.train(_trainDataFile=sentenciasFile, _validationSplit=validationSplit, _batch_size=batch_size, _num_epochs=num_epochs, _doTrainFlag=doTrainFlag, _doValidationFlag=doValidationFlag, _saveFlag=saveTraningFlag, _trainOnValidationData=trainOnValidationData)
    # ---- PROCESAR ----
    if(not(path.exists(modelFile)) and not(train_network)):
        print("WARNING: Using untrained model...")
    final_retun_obj = []
    for file_ in listdir(processFolder):
        print(">  Procesando archivo: {}".format(file_))
        filePath = path.join(processFolder,file_)
        aux_dict = {"path":filePath}
        aux_dict.update(DICTUM_core.extract_info_all(filePath,palabras_clave=palabras_clave,greek_num=greek_num))
        # DICTUM_utils.print_page(filePath,55)
        # input(file_)
        #find_text("ley", pdfPath = filePath)
        _, aux_ = dictum_model.predict(filePath, printResult_flag=True, _method= "batch.average", _batch_size=batch_size)
        aux_dict.update(aux_)
        final_retun_obj.append(aux_dict)
    # ---- GUARDAR ----
    DICTUM_utils.saveJson(final_retun_obj,final_retun_path)
