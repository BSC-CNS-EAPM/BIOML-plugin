"""
A module that performs regression analysis on a dataset.
"""

import os

from HorusAPI import (
    PluginVariable,
    SlurmBlock,
    VariableGroup,
    VariableList,
    VariableTypes,
    Extensions
)

# ==========================#
# Classifcal Variable inputs
# ==========================#
fastaFile = PluginVariable(
    name="Fasta file",
    id="fasta_file",
    description="The fasta file path.",
    type=VariableTypes.FILE,
    defaultValue=None,
    allowedValues=["fasta", "fsa"],
)
modelPath = PluginVariable(
    name="Model Path",
    id="model_path",
    description="The path to the model folder",
    type=VariableTypes.FOLDER,
    defaultValue=None,
)
testFeatures = PluginVariable(
    name="Test Features",
    id="test_features",
    description="The file to where the test features are saved in excel or csv format.",
    type=VariableTypes.FILE,
    defaultValue=None,
    allowedValues=["csv", "xlsx"],
)
trainingFeatures = PluginVariable(
    name="Training Features",
    id="training_features",
    description="The file to where the training features are saved in excel or csv format.",
    type=VariableTypes.FILE,
    defaultValue=None,
    allowedValues=["csv", "xlsx"],
)

Problem = PluginVariable(
    name="classification or regression",
    id="problem",
    description="The machine learning problem: classification or regression",
    type=VariableTypes.STRING,
    allowedValues=["classification", "regression"],
)

inputLables = PluginVariable(
    name="Input labels",
    id="in_labels",
    description="The label column name if label in features to remove it, otherwise unnecessary",
    type=VariableTypes.STRING,
)

### ==========================#
#   Finetuning Variables
### ==========================#


peftModel = PluginVariable(
    name="Peft model",
    id="peft_model",
    description="Path to the peft model to use for the prediction",
    type=VariableTypes.FOLDER,
    defaultValue=None,
)

LLMConfig = PluginVariable(
    name="LLM Config",
    id="llm_config",
    description="The config file to use for LLM model in json or yaml format",
    type=VariableTypes.FILE,
    defaultValue=None,
    allowedValues=["json", "yaml"],
)

TokenizerConfig = PluginVariable(
    name="Tokenizer Config",
    id="tokenizer_config",
    description="The config file to use for Tokenizer in json or yaml format",
    type=VariableTypes.FILE,
    defaultValue=None,
    allowedValues=["json", "yaml"],
)


### ==========================#
# Variable groups
### ==========================#

FineTuneGroup = VariableGroup(
    id="fine_tune_group",
    name="Fine Tuning",
    description="Using the fine tuned model for prediction",
    variables=[Problem, peftModel]
)

ClassicalGroup = VariableGroup(
    id="classical_group",
    name="Classical ML",
    description="Using the classical ML model for prediction",
    variables=[Problem, modelPath, testFeatures],
)

# ==========================#
# Variable outputs
# ==========================#
outputPrediction = PluginVariable(
    name="Prediction output",
    id="out_zip",
    description="The zip file to the output for the prediction results",
    type=VariableTypes.FILE,
)

##############################
#       Other variables      #
##############################


predictionOutput = PluginVariable(
    name="Prediction Output",
    id="prediction_dir",
    description="The path where to save the prediction results.",
    type=VariableTypes.STRING,
    defaultValue="prediction_results/predictions.csv",
)

numThreads = PluginVariable(
    name="Number of threads",
    id="num_threads",
    description="The number of threads to use.",
    type=VariableTypes.INTEGER,
    defaultValue=20,
)

scalerVar = PluginVariable(
    name="Scaler",
    id="scaler",
    description="Choose one of the scaler available in scikit-learn, defaults to zscore.",
    type=VariableTypes.STRING_LIST,
    allowedValues=["robust", "zscore", "minmax"],
    defaultValue="zscore",
)

outliersTrain = PluginVariable(
    name="Outliers train",
    id="outliers_train",
    description="Path to a list of outliers if any a file in plain text format, each record should be in a new line, the name should be the same as in the excel file with the filtered features",
    type=VariableTypes.FILE,
    defaultValue=None,
)

outliersTest = PluginVariable(
    name="Outliers test",
    id="outliers_test",
    description="Path to a list of outliers if any a file in plain text format, each record should be in a new line, the name should be the same as in the excel file with the filtered features",
    type=VariableTypes.FILE,
    defaultValue=None,
)


testSheetName = PluginVariable(
    name="Test Sheet Name",
    id="test_sheet_name",
    description="The sheet name for the excel file if the test features is in excel format.",
    type=VariableTypes.STRING,
    defaultValue=None,
)

sheetName = PluginVariable(
    name="Sheet Name",
    id="sheet_name",
    description="The sheet name for the excel file if the training features is in excel format.",
    type=VariableTypes.STRING,
    defaultValue=None,
)

numSimilarSamples = PluginVariable(
    name="Number of similar samples",
    id="num_similar_samples",
    description="The number of similar samples to use for filtering based on applicability domain",
    type=VariableTypes.INTEGER,
    defaultValue=1,
)

applicabilityDomain = PluginVariable(
    name="applicability domain",
    id="applicability_domain",
    description="If to use applicability domain filtering",
    type=VariableTypes.BOOLEAN,
    defaultValue=False,
)


def runPredictionBioml(block: SlurmBlock):
    from pathlib import Path

    ## Other variables for classical
    scaler = block.variables.get("scaler", "zscore")
    sheets = block.variables.get("sheet_name", None)
    test_sheets = block.variables.get("test_sheet_name", None)
    num_similar_samples = block.variables.get("num_similar_samples", 1)
    applicability_domain = block.variables.get("applicability_domain", False)
    outliers_train = block.variables.get("outliers_train", None)
    outliers_test = block.variables.get("outliers_test", None)
    input_label = block.variables.get("in_labels", None)
    training_features = block.variables.get("training_features", None)
    ## Other variables for fine tuning
    llm_config = block.variables.get("llm_config", None)
    tokenizer_args = block.variables.get("tokenizer_config", None)

    ## Shared variables
    prediction_output = block.variables.get("prediction_dir", "prediction_results/predictions.csv")
    num_threads = block.variables.get("num_threads", 20)
    input_fasta = block.variables.get("fasta_file", None)
    folderName = "prediction_inputs"

    #inputs
    problem = block.inputs.get("problem", None)
    if problem is None:
        raise Exception("No problem provided")

    if block.selectedInputGroup == "fine_tune_group":
        pef_model = block.inputs.get("peft_model", None)
        if input_fasta is None:
            raise Exception("No input fasta provided, required for the fine tuned model predictions")
        if pef_model:
            os.system(f"cp -r {pef_model} {folderName}")

    elif block.selectedInputGroup == "classical_group":
        test_features = block.inputs.get("test_features", None)
        if test_features is None:
            raise Exception("No test features provided")
        if not os.path.exists(test_features):
            raise Exception(f"The test features file does not exist: {test_features}")        

        model_path = block.inputs.get("model_path", None)
        if model_path is None:
            raise Exception("No model path provided")
        if not os.path.exists(model_path):
            raise Exception(f"The model path does not exist: {model_path}")

        if input_fasta is None and problem == "classification" and applicability_domain:
            raise Exception("No fasta file provided, it is required for classification in classical ML")
        if input_fasta and not os.path.exists(input_fasta):
            raise Exception(f"The input fasta file does not exist: {input_fasta}")
        
        if test_features.endswith(".xlsx") and not test_sheets:
            raise Exception(
                "The test features is in excel format, please provide the sheet name"
            )
        
        os.system(f"cp {test_features} {folderName}")
        os.system(f"cp -r {model_path} {folderName}")

    if training_features:
        if not os.path.exists(training_features):
            raise Exception(
                f"The training features file does not exist: {training_features}"
            )
        if training_features.endswith(".xlsx") and not sheets:
            raise Exception(
                "The training features is in excel format, please provide the sheet name"
            )

    # Create an copy the inputs
    os.makedirs(folderName, exist_ok=True)
    if input_fasta:
        os.system(f"cp {input_fasta} {folderName}")
    os.system(f"cp {training_features} {folderName}")

    if outliers_train and not os.path.exists(outliers_train):
            raise Exception(f"The outliers train file does not exist: {outliers_train}")
    os.system(f"cp {outliers_train} {folderName}")
    if outliers_test and not os.path.exists(outliers_test):
            raise Exception(f"The outliers test file does not exist: {outliers_test}")
    os.system(f"cp {outliers_test} {folderName}")


    if llm_config:
        os.system(f"cp {llm_config} {folderName}")
    if tokenizer_args:
        os.system(f"cp {tokenizer_args} {folderName}")
    # Run the command
    command = "python -m BioML.models.predict "
    if input_fasta:
        command += f"--fasta_file {folderName}/{Path(input_fasta).name} "

    if block.selectedInputGroup == "classical_group":
        command += f"--model_path {folderName}/{Path(model_path).name} "
        command += f"--test_features {folderName}/{Path(test_features).name} "
    elif block.selectedInputGroup == "fine_tune_group":
        command += f"-d "
        command += f"--peft {folderName}/{Path(pef_model).name} "
        if llm_config:
            command += f"-lc {llm_config} "
        if tokenizer_args:
            command += f"-tc {tokenizer_args} "

    if training_features:
        command += f"--training_features {folderName}/{Path(training_features).name} "

    command += f"--problem {problem} "
    command += f"--scaler {scaler} "
    command += f"-nss {num_similar_samples} "
    if  applicability_domain:
        command += f"-ad {applicability_domain} "
    if sheets:
        command += f"--sheets {sheets} "
    if input_label:
        command += f"--label {input_label} "
    if outliers_test:
        command += f"--outliers_test {folderName}/{Path(outliers_test).name} "
    if outliers_train:
        command += f"--outliers_train {folderName}/{Path(outliers_train).name} "
    if test_sheets:
        command += f"-ts {test_sheets} "

    command += f"--res_dir {prediction_output} "

    jobs = [command]


    block.extraData["prediction_dir"] = prediction_output
    block.variables["cpus"] = num_threads
    
    from utils import launchCalculationAction

    launchCalculationAction(
        block,
        jobs,
        program="bioml",
        uploadFolders=[
            folderName,
        ],
    )


def finalAction(block: SlurmBlock):
    from utils import downloadResultsAction
    from pathlib import Path
    e = Extensions()

    downloaded_path = downloadResultsAction(block)
    prediction_output = block.extraData.get("prediction_dir", "prediction_result/predictions.csv")

    block.setOutput(outputPrediction.id, Path(downloaded_path)/prediction_output)
    e.loadCSV(
        str(Path(downloaded_path)/prediction_output),
        "predictions",
    )

from utils import BSC_JOB_VARIABLES

PredictBlock = SlurmBlock(
    name="Predict",
    id="predict",
    initialAction=runPredictionBioml,
    finalAction=finalAction,
    description="Predict using the models and average the votations.",
    inputGroups=[
        FineTuneGroup,
        ClassicalGroup,
    ],
    variables=BSC_JOB_VARIABLES + [
        scalerVar,
        sheetName,
        inputLables,
        numSimilarSamples,
        applicabilityDomain,
        outliersTrain,
        outliersTest,
        predictionOutput,
        fastaFile, 
        testSheetName,
        numThreads,
        LLMConfig, TokenizerConfig, 
        trainingFeatures,

    ],
    outputs=[outputPrediction],
)
