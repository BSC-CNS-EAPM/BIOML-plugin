"""
Bioml Classification
    | Wrapper class for the bioml Classification module.
    | Train classification models.
"""

# TODO Add to the documentation

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
inputLabelFile = PluginVariable(
    name="Input Label File",
    id="input_label_file",
    description="The path to the labels of the training set in a csv format or string if it is inside training features.",
    type=VariableTypes.FILE,
    defaultValue=None,
    allowedValues=["csv"],
)
inputLabelString = PluginVariable(
    name="Input Label String",
    id="input_label_string",
    description="The labels of the training set in a string format.",
    type=VariableTypes.STRING,
    defaultValue=None,
)
trainingFeatures = PluginVariable(
    name="Training Features",
    id="training_features",
    description="The file to where the training features are saved in excel or csv format.",
    type=VariableTypes.FILE,
    defaultValue=None,
    allowedValues=["csv", "xlsx"],
)

tuneVar = PluginVariable(
    name="Tune",
    id="tune",
    description="If to tune the best models.",
    type=VariableTypes.BOOLEAN,
    defaultValue=True,
)

optimizeVar = PluginVariable(
    name="Optimize",
    id="optimize",
    description="The metric to optimize for retuning the best models.",
    type=VariableTypes.STRING_LIST,
    defaultValue="MCC",
    allowedValues=[
        "MCC",
        "Prec.",
        "Recall",
        "F1",
        "AUC",
        "Accuracy",
        "Average Precision Score",
    ],
)

fileGroup = VariableGroup(
    id="fileType_input",
    name="Input File",
    description="The input is a file",
    variables=[inputLabelFile, trainingFeatures, tuneVar, optimizeVar],
)
stringGroup = VariableGroup(
    id="stringType_input",
    name="Input String",
    description="The input is a string",
    variables=[inputLabelString, trainingFeatures, tuneVar, optimizeVar],
)

# ==========================#
# Variable outputs
# ==========================#
outputClassification = PluginVariable(
    name="Classification output",
    id="out_zip",
    description="The zip file to the output for the classification models",
    type=VariableTypes.FILE,
)

##############################
#       Other variables      #
##############################
trainingOutput = PluginVariable(
    name="Training Output",
    id="training_output",
    description="The path where to save the models training results.",
    type=VariableTypes.STRING,
    defaultValue="classification_results",
)
scalerVar = PluginVariable(
    name="Scaler",
    id="scaler",
    description="Choose one of the scaler available in scikit-learn, defaults to zscore.",
    type=VariableTypes.STRING_LIST,
    allowedValues=["robust", "zscore", "minmax"],
    defaultValue="zscore",
)
kfoldParameters = PluginVariable(
    name="Kfold Parameters",
    id="kfold_parameters",
    description="The parameters for the kfold in num_split:test_size format ('5:0.2').",
    type=VariableTypes.STRING,
    defaultValue="5:0.2",
)
outliersVar = PluginVariable(
    name="Outliers",
    id="outliers",
    description="Path to the outlier file in plain text format, the name should be the same as in the excel file with the filtered features, each record should be in a new line",
    type=VariableTypes.FILE,
    defaultValue=None,
)
budgetTime = PluginVariable(
    name="Budget Time",
    id="budget_time",
    description="The time budget for the training in minutes, should be > 0 or None.",
    type=VariableTypes.FLOAT,
    defaultValue=None,
)
precisionWeight = PluginVariable(
    name="Precision Weight",
    id="precision_weight",
    description="Weights to specify how relevant is the precision for the ranking of the different features.",
    type=VariableTypes.FLOAT,
    defaultValue=1.2,
)
recallWeight = PluginVariable(
    name="Recall Weight",
    id="recall_weight",
    description="Weights to specify how relevant is the recall for the ranking of the different features.",
    type=VariableTypes.FLOAT,
    defaultValue=0.8,
)
reportWeight = PluginVariable(
    name="Report Weight",
    id="report_weight",
    description="Weights to specify how relevant is the f1, precision and recall for the ranking of the different features with respect to MCC which is a more general measures of the performance of a model.",
    type=VariableTypes.FLOAT,
    defaultValue=1,
)
differenceWeight = PluginVariable(
    name="Difference Weight",
    id="difference_weight",
    description="How important is to have similar training and test metrics.",
    type=VariableTypes.FLOAT,
    defaultValue=1.2,
)
bestModels = PluginVariable(
    name="Best Models",
    id="best_models",
    description="The number of best models to select, it affects the analysis and the saved hyperparameters.",
    type=VariableTypes.INTEGER,
    defaultValue=3,
)
seedVar = PluginVariable(
    name="Seed",
    id="seed",
    description="The seed for the random state.",
    type=VariableTypes.INTEGER,
    defaultValue=63462634,
)
dropVar = PluginVariable(
    name="Drop",
    id="drop",
    description="The models to drop.",
    type=VariableTypes.CHECKBOX,
    defaultValue=["ada"],
    allowedValues=[
        "lr",
        "knn",
        "nb",
        "dt",
        "svm",
        "rbfsvm",
        "gpc",
        "mlp",
        "ridge",
        "rf",
        "qda",
        "ada",
        "gbc",
        "lda",
        "et",
        "xgboost",
        "lightgbm",
        "catboost",
        "dummy",
    ],
)
selectedVar = PluginVariable(
    name="Selected",
    id="selected",
    description="The only models to train",
    type=VariableTypes.CHECKBOX,
    defaultValue=None,
    allowedValues=[
        "lr",
        "knn",
        "nb",
        "dt",
        "svm",
        "rbfsvm",
        "gpc",
        "mlp",
        "ridge",
        "rf",
        "qda",
        "ada",
        "gbc",
        "lda",
        "et",
        "xgboost",
        "lightgbm",
        "catboost",
        "dummy",
    ],
)

plotVar = PluginVariable(
    name="Plot",
    id="plot",
    description="The plots to save.",
    type=VariableTypes.CHECKBOX,
    defaultValue=["learning", "confusion_matrix", "class_report"],
    allowedValues=["learning", "confusion_matrix", "class_report", "pr", "auc"],
)

sheetName = PluginVariable(
    name="Sheet Name",
    id="sheet_name",
    description="The sheet name for the excel file if the training features is in excel format.",
    type=VariableTypes.STRING,
    defaultValue=None,
)
numIter = PluginVariable(
    name="Number of Iterations",
    id="num_iter",
    description="The number of iterations for the hyperparameter search.",
    type=VariableTypes.INTEGER,
    defaultValue=30,
)

numThreads = PluginVariable(
    name="Number of threads",
    id="num_threads",
    description="The number of threads to use.",
    type=VariableTypes.INTEGER,
    defaultValue=100,
)

splitStrategy = PluginVariable(
    name="Split Strategy",
    id="split_strategy",
    description="The strategy to split the data.",
    type=VariableTypes.STRING_LIST,
    defaultValue="stratifiedkfold",
    allowedValues=["mutations", "cluster", "stratifiedkfold", "kfold"],
)
clusterVar = PluginVariable(
    name="Cluster",
    id="cluster",
    description="The path to the cluster file generated by mmseqs2 or a custom group index file just like data/resultsDB_clu.tsv.",
    type=VariableTypes.FILE,
    defaultValue=None,
)
mutationsVar = PluginVariable(
    name="Mutations",
    id="mutations",
    description="The column name of the mutations in the training data.",
    type=VariableTypes.STRING,
    defaultValue=None,
)
testNumMutations = PluginVariable(
    name="Test Number of Mutations",
    id="test_num_mutations",
    description="The threshold of number of mutations to be included in the test set.",
    type=VariableTypes.INTEGER,
    defaultValue=None,
)
greaterVar = PluginVariable(
    name="Greater",
    id="greater",
    description="Include in the test set, mutations that are greater of less than the threshold, default greater.",
    type=VariableTypes.BOOLEAN,
    defaultValue=True,
)
shuffleVar = PluginVariable(
    name="Shuffle",
    id="shuffle",
    description="If to shuffle the data before splitting.",
    type=VariableTypes.BOOLEAN,
    defaultValue=True,
)
crossValidation = PluginVariable(
    name="Cross Validation",
    id="cross_validation",
    description="If to use cross validation, default is True.",
    type=VariableTypes.BOOLEAN,
    defaultValue=True,
)


def runClassificationBioml(block: SlurmBlock):
    from pathlib import Path
    #inputs
    training_features = block.inputs.get("training_features", None)
    if training_features is None:
        raise Exception("No input features provided")
    if not os.path.exists(training_features):
        raise Exception(f"The input features file does not exist: {training_features}")

    optimize = block.inputs.get("optimize", "MCC")
    tune = block.inputs.get("tune", True)
    
    input_label_string = block.inputs.get("input_label_string", None)
    input_label_file = block.inputs.get("input_label_file", None)
    
    if input_label_file is None and input_label_string is None:
        raise Exception("No input label provided")
    if not os.path.exists(input_label_file):
        raise Exception(f"The input label file does not exist: {input_label}")

    ## other varibales
    sheets = block.variables.get("sheet_name", None)
    seed = block.variables.get("seed", 63462634)
    num_Iter = block.variables.get("num_iter", 30)
    split_strategy = block.variables.get("split_strategy", "stratifiedkfold")
    cluster = block.variables.get("cluster", None)
    shuffle = block.variables.get("shuffle", True)
    cross_validation = block.variables.get("cross_validation", True)
    mutations = block.variables.get("mutations", None)
    test_Num_Mutations = block.variables.get("test_num_mutations", None)
    greater = block.variables.get("greater", True)
    scaler = block.variables.get("scaler", "zscore")
    kfold_parameters = block.variables.get("kfold_parameters", "5:0.2")
    outliers = block.variables.get("outliers", None)
    selected_models = block.variables.get("selected", None)
    plot = block.variables.get("plot", ["learning", "confusion_matrix", "class_report"])
    best_models = block.variables.get("best_models", 3)
    report_weight = block.variables.get("report_weight", 1)
    difference_weight = block.variables.get("difference_weight", 1.2)
    precision_weight = block.variables.get("precision_weight", 1.2)
    drop = block.variables.get("drop", ["ada"])
    budget_time = block.variables.get("budget_time", None)
    recall_weight = block.variables.get("recall_weight", 0.8)
    num_threads = block.variables.get("num_threads", 100)
    
    ## Create the output folder
    training_output = block.variables.get("training_output", "classification_results")
    block.extraData["output_folder"] = training_output
    # Create an copy the inputs
    folderName = "savemodel_inputs"
    os.makedirs(folderName, exist_ok=True)
    os.system(f"cp {input_label} {folderName}")
    os.system(f"cp {training_features} {folderName}")
    if cluster:
        if not os.path.exists(cluster):
            raise Exception(f"The cluster file does not exist: {cluster}")
        os.system(f"cp {cluster} {folderName}")
    if outliers:
        if not os.path.exists(outliers):
            raise Exception(f"The outliers file does not exist: {outliers}")
        os.system(f"cp {outliers} {folderName}")
    
    ## Command
    command = "python -m BioML.models.save_models "
    command += f"-l {folderName}/{input_label} "
    command += f"-i {folderName}/{training_features} "
    command += f"-sc {scaler} "
    command += f"-o {training_output} "
    command += f"-k {kfold_parameters} "
    command += f"--seed {seed} "
    command += f"-st {split_strategy} "
    command += f"-bu {budget_time} "
    command += f"-ni {num_Iter} "
    command += f"-rpw {report_weight} "
    command += f"-dw {difference_weight} "
    command += f"-be {best_models} "
    command += f"-op {optimize} "
    command += f"-pw {precision_weight} "
    command += f"-rw {recall_weight} "
    
    if plot:
        command += f"-p {' '.join(plot)} "
    if drop:
        command += f"-dr {' '.join(drop)} "
    if selected_models:
        command += f"-se {' '.join(selected_models)} "
    if not shuffle:
        command += f"-sf "
    if not cross_validation:
        command += f"-cv "
    if mutations:
        command += f"-m {mutations} "
    if test_Num_Mutations:
        command += f"-tnm {test_Num_Mutations} "
    if not greater:
        command += f"-g "
    if cluster:
        command += f"-c {folderName}/{Path(cluster).name} "
    if outliers:
        command += f"-ot {folderName}/{Path(outliers).name} "
    if not tune:
        command += f"--tune "
    if sheets:
        command += f"-sh {sheets} "

    jobs = [command]

    block.variables["script_name"] = "classification.sh"
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
    print("Classifications Models finished")

    from pathlib import Path
    from utils import downloadResultsAction

    download_path = downloadResultsAction(block)

    output_folder = block.extraData.get("output_folder", "classification_results")
    FolderName = Path(download_path) / output_folder

    block.setOutput(outputClassification.id, FolderName)


from utils import BSC_JOB_VARIABLES

classificationBlock = SlurmBlock(
    name="Classification BioML",
    initialAction=runClassificationBioml,
    finalAction=finalAction,
    description="Train classification models.",
    inputGroups=[fileGroup, stringGroup],
    variables=BSC_JOB_VARIABLES
    + [
        selectedVar,
        dropVar,
        trainingOutput,
        scalerVar,
        kfoldParameters,
        outliersVar,
        budgetTime,
        precisionWeight,
        recallWeight,
        reportWeight,
        differenceWeight,
        bestModels,
        seedVar,
        plotVar,
        sheetName,
        numIter,
        splitStrategy,
        clusterVar,
        mutationsVar,
        testNumMutations,
        greaterVar,
        shuffleVar,
        crossValidation,
        numThreads,
    ],
    outputs=[outputClassification],
)
