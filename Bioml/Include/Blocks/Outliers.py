"""
A module that performs outlier detection using features
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
excelFile = PluginVariable(
    name="Input features",
    id="input_file",
    description="The file to where the selected features are saved in excel or csv format.",
    type=VariableTypes.FILE,
    defaultValue=None,
    allowedValues=["csv", "xlsx"],
)


# ==========================#
# Variable outputs
# ==========================#
outputOutliers = PluginVariable(
    name="Outliers output",
    id="out_zip",
    description="The path to the output for the outliers.",
    type=VariableTypes.FILE,
)

outputCsv = PluginVariable(
    name="Outlier csv",
    id="out_csv",
    description="The path to the output csv file",
    type=VariableTypes.STRING,
    defaultValue="training_results/outliers.csv",
)

##############################
#       Other variables      #
##############################
numThreads = PluginVariable(
    name="Number of threads",
    id="num_threads",
    description="The number of threads to use.",
    type=VariableTypes.INTEGER,
    defaultValue=-1,
)
scalerVar = PluginVariable(
    name="Scaler",
    id="scaler",
    description="The scaler to use.",
    type=VariableTypes.STRING,
    defaultValue="robust",
    allowedValues=["robust", "standard", "minmax"],
)
contaminationVar = PluginVariable(
    name="Contamination",
    id="contamination",
    description="The contamination value.",
    type=VariableTypes.FLOAT,
    defaultValue=0.06,
)
numFeatures = PluginVariable(
    name="Number of features",
    id="num_features",
    description="The fraction of features to use.",
    type=VariableTypes.FLOAT,
    defaultValue=1.0,
)


def runOutliersBioml(block: SlurmBlock):
    
    input_excel = block.inputs.get("input_excel", None)
    if input_excel is None:
        raise Exception("No input excel provided")
    if not os.path.exists(input_excel):
        raise Exception(f"The input excel file does not exist: {input_excel}")
    input_excel = block.remote.sendData(input_excel, block.remote.workDir)

    ## variables
    num_threads = block.variables.get("num_threads", -1)
    num_features = block.variables.get("num_features", 1.0)
    contamination = block.variables.get("contamination", 0.06)
    scaler = block.variables.get("scaler", "zscore")
    output_csv = block.variables.get("output_csv", "training_results/outliers.csv")
    
    ## change bsc variables
    if num_threads > 1:
        block.variables["cpus"] = num_threads

    block.variables["script_name"] = "outliers.sh"
    
    command = "python -m BioML.utilities.outlier "
    command += f"-i {input_excel} "
    command += f"-c {contamination} "
    command += f"-s {scaler} "
    command += f"-n {num_threads} "
    command += f"-o {output_csv} "
    command += f"-nfe {num_features} "

    jobs = [command]
    

    from utils import launchCalculationAction

    launchCalculationAction(
        block,
        jobs,
        program="bioml",
        uploadFolders=[
        ],
    )


def finalAction(block: SlurmBlock):
    from utils import downloadResultsAction
    downloaded_path = downloadResultsAction(block)
    
    block.setOutput(outputOutliers.id, downloaded_path)


from utils import BSC_JOB_VARIABLES

outliersBlock = SlurmBlock(
    name="Outlier",
    id="outliers",
    initialAction=runOutliersBioml,
    finalAction=finalAction,
    description="Outlier detection.",
    inputs=[excelFile],
    variables=BSC_JOB_VARIABLES
    + [
        numThreads,
        scalerVar,
        contaminationVar,
        numFeatures,
        outputCsv
        
    ],
    outputs=[outputOutliers],
)
