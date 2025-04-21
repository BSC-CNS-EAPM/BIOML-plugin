"""
Bioml Classification
    | Wrapper class for the bioml Classification module.
    | Train classification models.
    | You will be able to visualize the training results and the plots
"""

# TODO Add to the documentation

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
# Variable inputs
# ==========================#
inputLabelFile = PluginVariable(
    name="Input Label File",
    id="input_label_file",
    description="The path to the labels of the training set in a csv format",
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
    variables=[inputLabelFile, trainingFeatures],
)
stringGroup = VariableGroup(
    id="stringType_input",
    name="Input String",
    description="The input is a string",
    variables=[inputLabelString, trainingFeatures],
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
stratifiedVar = PluginVariable(
    name="Stratified",
    id="stratified",
    description="If to use stratified sampling.",
    type=VariableTypes.BOOLEAN,
    defaultValue=True,
)

iterateFeatures = PluginVariable(
    name="Iterate Multiple Features",
    id="iterate_features",
    description="If to iterate over multiple features.",
    type=VariableTypes.BOOLEAN,
    defaultValue=False,
)

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

logExperiments = PluginVariable(
    name="Log Experiments",
    id="log_experiments",
    description="If to log the experiments.",
    type=VariableTypes.BOOLEAN,
    defaultValue=False,
)


majorityVar = PluginVariable(
    name="Majority",
    id="majority",
    description="If to train majority model",
    type=VariableTypes.BOOLEAN,
    defaultValue=True,
)

stackVar = PluginVariable(
    name="Stack",
    id="stack",
    description="If to train stack model",
    type=VariableTypes.BOOLEAN,
    defaultValue=True,
)


# ==========================
def runClassificationBioml(block: SlurmBlock):
    from pathlib import Path
    #inputs
    training_features = block.inputs.get("training_features", None)
    if training_features is None:
        raise Exception("No input features provided")
    if not os.path.exists(training_features):
        raise Exception(f"The input features file does not exist: {training_features}")

    optimize = block.variables.get("optimize", "MCC")
    tune = block.variables.get("tune", True)
    
    if block.selectedInputGroup == "stringType_input":
        input_label = block.inputs.get("input_label_string", None)
        file=False
    else:
        input_label = block.inputs.get("input_label_file", None)
        file = True

    if input_label is None:
        raise Exception("No input label provided")
    if file and not os.path.exists(input_label):
        raise Exception(f"The input label file does not exist, use the absolute path: {input_label}")

    ## other varibales
    stratified = block.variables.get("stratified", True)
    iterate_features = block.variables.get("iterate_features", False)
    if iterate_features and not training_features.endswith(".xlsx"):
        raise Exception(
            "The iterate features option is only available for excel files."
        )
    
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
    log_experiments = block.variables.get("log_experiments", False)
    stratified = block.variables.get("stratified", True)
    majority = block.variables.get("majority", True)
    stack = block.variables.get("stack", True)
    
    if iterate_features:
        log_experiments = False
    
    ## Create the output folder
    training_output = block.variables.get("training_output", "classification_results")
    block.extraData["output_folder"] = training_output
    block.extraData["iterate_features"] = iterate_features

    # Create an copy the inputs
    folderName = "classi_inputs"
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
    command = "python -m BioML.models.classification "
    if not file:
        command += f"-l {input_label} "
    else:
        command += f"-l {folderName}/{Path(input_label).name} "
    command += f"-i {folderName}/{Path(training_features).name} "
    command += f"-s {scaler} "
    command += f"-o {training_output} "
    command += f"-k {kfold_parameters} "
    command += f"--seed {seed} "
    command += f"-st {split_strategy} "
    if budget_time is not None:
        command += f"-bu {budget_time} "
    command += f"-ni {num_Iter} "
    command += f"-rpw {report_weight} "
    command += f"-dw {difference_weight} "
    command += f"-be {best_models} "
    command += f"-op {optimize} "
    command += f"-pw {precision_weight} "
    command += f"-rw {recall_weight} "

    if iterate_features:
        command += f"-it "
    if not stratified:
        command += f"-ti "
    if plot:
        command += f"-p {' '.join(plot)} "
    if drop:
        command += f"-d {' '.join(drop)} "
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
    if tune:
        command += f"--tune "
    if sheets:
        command += f"-sh {sheets} "
    if log_experiments:
        command += f"-log "
    if not majority:
        command += f"-mj "
    if not stack:
        command += f"-stck "
        
    jobs = command

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

    import pandas as pd
    from pathlib import Path
    from utils import downloadResultsAction

    download_path = downloadResultsAction(block)

    output_folder = block.extraData.get("output_folder", "classification_results")
    FolderName = Path(download_path) / output_folder
    iterate_features = block.extraData.get("iterate_features", False)

    e = Extensions()
    block.setOutput(outputClassification.id, FolderName)
    if iterate_features:
        # visualize the results of the top 3 feature sets (each feature set is a sheet in the excel file 
        # and will show the results of the top 3 models trained for that feature set)
        if os.path.exists(FolderName / "training_results.xlsx"):
            excel_sheets = pd.ExcelFile(FolderName / "training_results.xlsx").sheet_names
            training_features = pd.read_excel(
                FolderName / "training_results.xlsx", sheet_name=excel_sheets[:3], index_col=[0,1,2]
            )
            for num, (sheet, df) in enumerate(training_features.items()):
                df.to_csv(FolderName / f"training_results_{sheet}.csv")
                e.loadCSV(
                    str(FolderName / f"training_results_{sheet}.csv"),
                    f"top{num}_results_{sheet}",
                )
    else:
        # After visualizing the top feature set and selecting that feature set for futher comparisons and tuning
        # visualize the results of the top 3 models trained for that feature set and plot theiur performance
        if os.path.exists(FolderName / "model_plots"):
            model_plots = Path(FolderName / "model_plots").glob("*/*/*.png")

            for plot in model_plots:
                if "not_tuned" in str(plot) and  "Confusion" not in str(plot):
                    e.loadImage(str(plot), f"{plot.parents[1].name}_{plot.parent.name}_{plot.stem}")
                    
            results = pd.read_excel(
                FolderName / "not_tuned" / "training_results.xlsx", sheet_name=["train", "test_results"], index_col=[0,1,2]
            )
            for num, (sheet, df) in enumerate(results.items()):
                df.to_csv(FolderName / f"training_results_{sheet}.csv")
                e.loadCSV(
                    str(FolderName / f"training_results_{sheet}.csv"),
                    f"results_{sheet}",
                )
from utils import BSC_JOB_VARIABLES

classificationBlock = SlurmBlock(
    name="Classification",
    id="classification",
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
        numThreads, tuneVar, optimizeVar,
        iterateFeatures,
        stratifiedVar, logExperiments,
        majorityVar, stackVar
    ],
    outputs=[outputClassification],
)
