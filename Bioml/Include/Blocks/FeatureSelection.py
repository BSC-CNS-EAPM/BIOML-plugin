"""
Bioml Selection
    | Wrapper class for the bioml Classification module.
    | Train classification models.
"""

from HorusAPI import PluginVariable, SlurmBlock, VariableTypes

# ==========================#
# Variable inputs
# ==========================#
inputCsv = PluginVariable(
    name="Input CSV",
    id="in_csv",
    description="The CSV file with the input features",
    type=VariableTypes.FILE,
)

inputLabels = PluginVariable(
    name="Input labels",
    id="in_labels",
    description="The label file or column name if label already in features",
    type=VariableTypes.STRING,
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
outputSelection = PluginVariable(
    name="Feature Selection output",
    id="out_zip",
    description="The features extracted",
    type=VariableTypes.FILE,
)

excelSelection = PluginVariable(
    name="excel output",
    id="excel_out",
    description="The features extracted",
    type=VariableTypes.STRING,
    defaultValue="training_features/selected_features.xlsx",
)

##############################
#       Other variables      #
##############################

featureRange = PluginVariable(
    name="feature range",
    id="feature_range",
    description="""Specify the minimum and maximum of number of features in start:stop:step format or
                            a single integer. Start will default to num samples / 10, Stop will default 
                            to num samples / 2 and step will be (stop - step / 5)""",
    type=VariableTypes.STRING,
    defaultValue="none:none:none",
)

numThreads = PluginVariable(
    name="Number of threads",
    id="num_threads",
    description="The number of threads to use.",
    type=VariableTypes.INTEGER,
    defaultValue=100,
)

varianceThreshold = PluginVariable(
    name="variance threshold",
    id="variance_threshold",
    description="""The variance the feature has to have, 0 means that the comlun has the same value for all samples. 
    None to deactivate""",
    type=VariableTypes.FLOAT,
    defaultValue=0.0,
)

scalerVar = PluginVariable(
    name="Scaler",
    id="scaler",
    description="The scaler to use.",
    type=VariableTypes.STRING,
    defaultValue="robust",
    allowedValues=["robust", "standard", "minmax"],
)

outliersVar = PluginVariable(
    name="Outliers",
    id="outliers",
    description="Path to a file in plain text format, each record should be in a new line, the name should be the same as in the excel file with the filtered features",
    type=VariableTypes.FILE,
    defaultValue=None,
)

sheetName = PluginVariable(
    name="Sheet Name",
    id="sheet_name",
    description="The sheet name for the excel file if the training features is in excel format.",
    type=VariableTypes.STRING,
    defaultValue=None,
)

rfeSteps = PluginVariable(
    name="Number RFE steps",
    id="rfe_steps",
    description="The number of RFE steps to use.",
    type=VariableTypes.INTEGER,
    defaultValue=30,
)

Plot = PluginVariable(
    name="plot",
    id="plot",
    description="Default to true, plot the feature importance using shap",
    type=VariableTypes.BOOL,
    defaultValue=True,
)

plotNumFeatures = PluginVariable(
    name="number of features to plot",
    id="plot_num_features",
    description="The number of features to plot",
    type=VariableTypes.INTEGER,
    defaultValue=20,
)

Seed = PluginVariable(
    name="Seed",
    id="seed",
    description="The seed for the random state.",
    type=VariableTypes.INTEGER,
    defaultValue=978236392,
)

def initialAction(block: SlurmBlock):
    import os

    # Inputs
    input_csv = block.inputs.get("in_csv", None)
    if input_csv is None:
        raise Exception("No input csv provided")
    if not os.path.exists(input_csv):
        raise Exception(f"The input csv file does not exist: {input_csv}")
    
    input_csv = block.remote.sendData(input_csv, block.remote.workDir)

    input_label = block.inputs.get("in_labels", None)
    if input_label is None:
        raise Exception("No input label provided")
    if os.path.exists(input_label):
        input_label = block.remote.sendData(input_label, block.remote.workDir)
    
    problem = block.inputs.get("problem", None)
    if problem is None:
        raise Exception("No problem provided")
    
    # Variables
    feature_range = block.variables.get("featrure_range", "none:none:none")
    num_threads = block.variables.get("num_threads", 100)
    seed = block.variables.get("seed", 978236392)
    plot_num_features = block.variables.get("plot_num_features", 20)
    rfe_steps = block.variables.get("rfe_steps", 30)
    scaler = block.variables.get("scaler", "robust")
    variance_threshold = block.variables.get("variance_threshold", 0.0)
    plot = block.variables.get("plot", True)
    outliers = block.variables.get("outliers", None)
    sheet_name = block.variables.get("sheet_name", None)
    if outliers:
        if not os.path.exists(outliers):
            raise Exception(f"The outliers file does not exist: {outliers}")
        outliers = block.remote.sendData(outliers, block.remote.workDir)
    # Outputs
    excel_selection = block.variables.get("excel_out", "training_features/selected_features.xlsx")
    block.extraData["excel_selection"] = excel_selection

    command = "python -m BioML.features.selection "
    command += f"-f {input_csv} "
    command += f"-l {input_label} "
    command += f"-pr {problem} "
    command += f"-r {feature_range} "
    command += f"-v {variance_threshold} "
    if not plot:
        command += f"-p "
    command += f"-pk {plot_num_features} "
    command += f"-rt {rfe_steps} "
    command += f"-s {scaler} "
    command += f"-se {seed} "
    command += f"-n {num_threads} "
    command += f"-e {excel_selection} "
    if outliers:
        command += f"-ot {outliers} "
    if sheet_name:
        command += f"-sh {sheet_name} "

    jobs = [command]

    block.variables["cpus"] = num_threads
    block.variables["script_name"] = "feature_selection.sh"

    from utils import launchCalculationAction

    launchCalculationAction(
        block,
        jobs,
        program="bioml",
        uploadFolders=[
        ],
    )

def finalAction(block: SlurmBlock):
    from pathlib import Path
    from utils import downloadResultsAction
    downloaded_path = downloadResultsAction(block)
    excel_selection = block.extraData["excel_selection"]

    block.setOutput(outputSelection.id, Path(downloaded_path)/excel_selection)


from utils import BSC_JOB_VARIABLES

featureSelectionBlock = SlurmBlock(
    name="Feature Selection BioML",
    initialAction=initialAction,
    finalAction=finalAction,
    description="Feature Selection.",
    inputs=[inputCsv, Problem, inputLabels],
    variables=BSC_JOB_VARIABLES + [
        featureRange,
        numThreads,
        Seed,
        rfeSteps,
        scalerVar,
        varianceThreshold,
        Plot,
        plotNumFeatures,
        excelSelection 
    ],
    outputs=[outputSelection],
)
