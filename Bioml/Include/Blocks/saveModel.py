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
    description="The path to the labels of the training set in a csv format ",
    type=VariableTypes.FILE,
    defaultValue=None,
    allowedValues=["csv"],
)
inputLabelString = PluginVariable(
    name="Input Label String",
    id="input_label_string",
    description="The labels of the training set, if it is in the same file as the training features",
    type=VariableTypes.STRING,
    defaultValue=None,
)

selectedClassification = PluginVariable(
    name="Selected Models Classification",
    id="selected_models_classification",
    description="The classification models to train and save, it can be multiple, if it is an ensemble model",
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

selectedRegression = PluginVariable(
    name="Selected Models Regression",
    id="selected_models_regression",
    description="The regression models to train and save, it can be multiple, if it is an ensemble model",
    type=VariableTypes.CHECKBOX,
    defaultValue=None,
    allowedValues=['lr',
                'lasso',
                'ridge',
                'en',
                'lar',
                'llar',
                'omp',
                'br',
                'ard',
                'par',
                'ransac', 
                'tr',
                'huber',
                'kr',
                'svm',
                'knn',
                'dt',
                'rf',
                'et',
                'ada',
                'gbr',
                'mlp',
                'xgboost',
                'lightgbm',
                'catboost',
                'dummy'
    ],
)

optimizeClassifciation = PluginVariable(
    name="Optimize Classification",
    id="optimize classification",
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

optimizeRegression = PluginVariable(
    name="Optimize Regression",
    id="optimize regression",
    description="The metric to optimize for retuning the best models.",
    type=VariableTypes.STRING_LIST,
    defaultValue="RMSE",
    allowedValues=[
        "RMSE", "R2", "MSE", "MAE", "RMSLE", "MAPE"
    ],
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

Problem = PluginVariable(
    name="classification or regression",
    id="problem",
    description="The machine learning problem: classification or regression",
    type=VariableTypes.STRING,
    allowedValues=["classification", "regression"],
)

# ==========================#
# Variable outputs
# ==========================#
outputModel = PluginVariable(
    name="Model output",
    id="model_output",
    description="The saved model without the .pkl extension",
    type=VariableTypes.STRING,
)


fileGroup = VariableGroup(
    id="fileType_input",
    name="Input File",
    description="The input is a file",
    variables=[inputLabelFile, trainingFeatures, Problem],
)
stringGroup = VariableGroup(
    id="stringType_input",
    name="Input String",
    description="The input is a string",
    variables=[inputLabelString, trainingFeatures, Problem],
)

##############################
#       Other variables      #
##############################
ModelName = PluginVariable(
    name="Model file name",
    id="model_name",
    description="The filename for the model without any extensions",
    type=VariableTypes.STRING,
    defaultValue="models/trained_model",
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
kfoldParameters = PluginVariable(
    name="Kfold Parameters",
    id="kfold_parameters",
    description="The parameters for the kfold in num_split:test_size format ('5:0.2').",
    type=VariableTypes.STRING,
    defaultValue="5:0.2",
)

seedVar = PluginVariable(
    name="Seed",
    id="seed",
    description="The seed for the random state.",
    type=VariableTypes.INTEGER,
    defaultValue=63462634,
)

outliersVar = PluginVariable(
    name="Outliers",
    id="outliers",
    description="Path to a file in plain text format, each record should be in a new line, the name should be the same as in the excel file with the filtered features",
    type=VariableTypes.FILE,
    defaultValue=None,
)
ModelStrategy = PluginVariable(
    name="Model Strategy",
    id="model_strategy",
    description="The strategy to select the best models. majority, stacking or simple:0 (single models), the 0 indicate the index in the list of models. The index can be up to best_models -1.",
    type=VariableTypes.STRING,
    defaultValue="simple:0",
)

bestModels = PluginVariable(
    name="Best Models",
    id="best_models",
    description="The number of best models to select, it affects the analysis and the saved hyperparameters.",
    type=VariableTypes.INTEGER,
    defaultValue=3,
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
    description="The number of iterations for the hyperparameter search in retuning models.",
    type=VariableTypes.INTEGER,
    defaultValue=30,
)
splitStrategy = PluginVariable(
    name="Split Strategy",
    id="split_strategy",
    description="The strategy to split the data.",
    type=VariableTypes.STRING_LIST,
    defaultValue="cluster",
    allowedValues=["mutations", "cluster", "stratifiedkfold", "kfold"],
)
clusterVar = PluginVariable(
    name="Cluster",
    id="cluster",
    description="The path to the cluster file generated by mmseqs2 or a custom group index file just like data/resultsDB_clu.tsv.",
    type=VariableTypes.FILE,
    defaultValue=None,
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


def runSaveModelBioml(block: SlurmBlock):
    from pathlib import Path
    ## inputs
    problem = block.inputs.get("problem", None)

    if problem == "classification":
        selected_models = block.variables.get("selected_models_classification", None)
        optimize = block.variables.get("optimize_classification", "MCC")

    elif problem == "regression":
        selected_models = block.variables.get("selected_models_regression", None)
        optimize = block.variables.get("optimize_regression", "RMSE")

    else:
        raise Exception("No problem provided")
    
    if not selected_models:
        raise Exception("No selected models for the problem provided")
    
    if block.selectedInputGroup == "stringType_input":
        input_label = block.inputs.get("input_label_string", None)
        file = False
    else:
        input_label = block.inputs.get("input_label_file", None)
        file = True

    if input_label is None:
        raise Exception("No input label provided")
    if file and not os.path.exists(input_label):
        raise Exception(f"The input label file does not exist, use the absolute path: {input_label}")

    tune = block.variables.get("tune", True)
    
    training_features = block.inputs.get("training_features", None)
    sheets = block.variables.get("sheet_name", None)
    if training_features is None:
        raise Exception("No training features provided")
    if not os.path.exists(training_features):
        raise Exception(f"The training features file does not exist: {training_features}")

    if training_features.endswith(".xlsx") and not sheets:
            raise Exception("No sheet name provided for the excel_file")

    ## other variables
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
    model_strategy = block.variables.get("model_strategy", "simple:0")
    num_threads = block.variables.get("num_threads", 20)
    best_models = block.variables.get("best_models", 3)
    
    # outputs
    model_output = block.variables.get("model_name", "models/trained_model")

    # Create an copy the inputs
    folderName = "savemodel_inputs"
    os.makedirs(folderName, exist_ok=True)
    if file:
        os.system(f"cp {input_label} {folderName}")
    os.system(f"cp {training_features} {folderName}")
    if cluster and  not os.path.exists(cluster):
            raise Exception(f"The cluster file does not exist: {cluster}")
    os.system(f"cp {cluster} {folderName}")
    if outliers and not os.path.exists(outliers):
            raise Exception(f"The outliers file does not exist: {outliers}")
    os.system(f"cp {outliers} {folderName}")
    
    ## Command
    command = "python -m BioML.models.save_model "
    if file:
        command += f"-l {folderName}/{Path(input_label).name} "
    else:
        command += f"-l {input_label} "
    command += f"-i {folderName}/{Path(training_features).name} "
    command += f"-s {scaler} "
    command += f"-o {model_output} "
    command += f"-k {kfold_parameters} "
    command += f"--seed {seed} "
    command += f"-st {split_strategy} "
    command += f"-ms {model_strategy} "
    command += f"-ni {num_Iter} "
    command += f"-se {' '.join(selected_models)} "
    command += f"-op {optimize} "
    command += f"-p {problem} "
    command += f"-be {best_models} "

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

    jobs = command

    block.variables["script_name"] = "save_model.sh"
    block.variables["cpus"] = num_threads
    block.extraData["model_output"] = model_output
    

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
    print("Savd Models finished")
    from pathlib import Path
    from utils import downloadResultsAction

    downloaded_path = downloadResultsAction(block)

    folder_name = block.extraData.get("model_output", "models/trained_model")
    model = list((downloaded_path/Path(folder_name).parent).glob("*.pkl"))
    if len(model) == 1:
        model = model[0]
        block.setOutput(outputModel.id, f"{model.parent}/{model.stem}")
    elif len(model) == 0:
        raise Exception(f"No model found: {model}, please check the output folder: {downloaded_path}/{Path(folder_name).parent}")
    else:
        print(f"There are more than one model, please check the output folder: {downloaded_path}/{Path(folder_name).parent}")
        print(f"Setting the output to the first model: {model[0]}")
        model = model[0]
        block.setOutput(outputModel.id, f"{model.parent}/{model.stem}")


from utils import BSC_JOB_VARIABLES

SaveModelBlock = SlurmBlock(
    name="Save Models",
    id="save_models",
    initialAction=runSaveModelBioml,
    finalAction=finalAction,
    description="Save models.",
    inputGroups=[fileGroup, stringGroup],
    variables=BSC_JOB_VARIABLES
    + [ # Add the variables here
        ModelName,
        scalerVar,
        kfoldParameters,
        outliersVar,
        seedVar,
        sheetName,
        numIter,
        splitStrategy,
        clusterVar,
        mutationsVar,
        testNumMutations,
        greaterVar,
        shuffleVar,
        crossValidation,
        numThreads, tuneVar,
        selectedRegression,
        selectedClassification,
        optimizeClassifciation,
        optimizeRegression,
        bestModels,
        ModelStrategy,
    ],
    outputs=[outputModel],
)
