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
)

# ==========================#
# Variable inputs
# ==========================#
fastaFile = PluginVariable(
    name="Fasta file",
    id="fasta_file",
    description="The fasta file path.",
    type=VariableTypes.FILE,
    defaultValue=None,
    allowedValues=["fasta"],
)
modelPath = PluginVariable(
    name="Model Path",
    id="model_path",
    description="The path to the model folder",
    type=VariableTypes.Folder,
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
    name="The machine learning problem",
    id="problem",
    description="The machine learning problem: classification or regression",
    type=VariableTypes.STRING_LIST,
    allowedValues=["classification", "regression"],
)

# ==========================#
# Variable outputs
# ==========================#
outputPrediction = PluginVariable(
    name="Prediction output",
    id="out_zip",
    description="The zip file to the output for the prediction results",
    type=VariableTypes.FOLDER,
)

##############################
#       Other variables      #
##############################
predictionOutput = PluginVariable(
    name="Prediction Output",
    id="prediction_dir",
    description="The path where to save the prediction results.",
    type=VariableTypes.STRING,
    defaultValue="prediction_results",
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
    type=VariableTypes.STRING,
    defaultValue=None,
)

outliersTest = PluginVariable(
    name="Outliers test",
    id="outliers_test",
    description="Path to a list of outliers if any a file in plain text format, each record should be in a new line, the name should be the same as in the excel file with the filtered features",
    type=VariableTypes.STRING,
    defaultValue=None,
)

inputLables = PluginVariable(
    name="Input labels",
    id="in_labels",
    description="The label  column name if label in features to remove it, otherwise unnecessary",
    type=VariableTypes.STRING,
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
    defaultValue=True,
)


def runPredictionBioml(block: SlurmBlock):
    from pathlib import Path

    #inputs
    input_fasta = block.inputs.get("fasta_file", None)
    if input_fasta is None:
        raise Exception("No input fasta provided")
    if not os.path.exists(input_fasta):
        raise Exception(f"The input fasta file does not exist: {input_fasta}")
    training_features = block.inputs.get("training_features", None)
    if training_features is None:
        raise Exception("No training features provided")
    if not os.path.exists(training_features):
        raise Exception(
            f"The training features file does not exist: {training_features}"
        )

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
    
    problem = block.inputs.get("problem", None)
    if problem is None:
        raise Exception("No problem provided")

    ## Other variables
    scaler = block.variables.get("scaler", "zscore")
    sheets = block.variables.get("sheet_name", None)
    label_name = block.variables.get("in_label", None)
    num_similar_samples = block.variables.get("num_similar_samples", 1)
    applicability_domain = block.variables.get("applicability_domain", True)
    outliers_train = block.variables.get("outliers_train", None)
    outliers_test = block.variables.get("outliers_test", None)
    prediction_output = block.variables.get("prediction_dir", "prediction_results")
    
    folderName = "prediction_inputs"

    # Create an copy the inputs
    os.makedirs(folderName, exist_ok=True)
    os.system(f"cp {input_fasta} {folderName}")
    os.system(f"cp {training_features} {folderName}")
    os.system(f"cp {test_features} {folderName}")
    os.system(f"cp -r {model_path} {folderName}")
    if outliers_train:
        if not os.path.exists(outliers_train):
            raise Exception(f"The outliers train file does not exist: {outliers_train}")
        os.system(f"cp {outliers_train} {folderName}")
    if outliers_test:
        if not os.path.exists(outliers_test):
            raise Exception(f"The outliers test file does not exist: {outliers_test}")
        os.system(f"cp {outliers_test} {folderName}")

    # Run the command
    command = "python -m BioML.models.predict "
    command += f"--fasta_file {folderName}/{Path(input_fasta).name} "
    command += f"--model_path {folderName}/{Path(model_path).name} "
    command += f"--training_features {folderName}/{Path(training_features).name} "
    command += f"--test_features {folderName}/{Path(test_features).name} "
    command += f"--problem {problem} "
    command += f"--scaler {scaler} "
    command += f"-nss {num_similar_samples} "
    if not applicability_domain:
        command += f"-ad {applicability_domain} "
    if sheets:
        command += f"--sheets {sheets} "
    if label_name:
        command += f"--label {label_name} "
    if outliers_test:
        command += f"--outliers_test {folderName}/{Path(outliers_train).name} "
    if outliers_train:
        command += f"--outliers_train {folderName}/{Path(outliers_test).name} "

    command += f"--res_dir {prediction_output} "

    jobs = [command]


    block.extraData["prediction_dir"] = prediction_output

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

    downloaded_path = downloadResultsAction(block)
    prediction_output = block.extraData.get("prediction_dir", "prediction_results")

    block.setOutput(outputPrediction.id, Path(downloaded_path)/prediction_output)


from utils import BSC_JOB_VARIABLES

PredictBlock = SlurmBlock(
    name="Predict BioMl",
    initialAction=runPredictionBioml,
    finalAction=finalAction,
    description="Predict using the models and average the votations.",
    inputs=[fastaFile, modelPath, testFeatures, trainingFeatures, Problem],
    variables=BSC_JOB_VARIABLES + [
        scalerVar,
        sheetName,
        inputLables,
        numSimilarSamples,
        applicabilityDomain,
        outliersTrain,
        outliersTest,
        predictionOutput,
        
    ],
    outputs=[outputPrediction],
)
