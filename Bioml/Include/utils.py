# TODO needs a database

from HorusAPI import PluginBlock, PluginVariable, SlurmBlock, VariableList, VariableTypes
import datetime
import os
import shutil
import subprocess
import typing
from pathlib import Path

# TODO adaptar a los dos tipos de extraction
def extract_features(
    minimal_sequence_length=100,
    extraction_type="ifeature",
    fasta_file=None,
    ifeature_path="/home/lavane/Repos/iFeature",
    possum_path="/home/lavane/sdb/Programs/POSSUM_Standalone_Toolkit",
):
    from BioML.features.extraction import (
        IfeatureFeatures,
        PossumFeatures,
        read_features,
    )
    from BioML.utilities.utils import clean_fasta

    clean_fasta(
        possum_path,
        fasta_file,
        "cleaned.fasta",
        minimal_sequence_length,
    )
    fasta_file = "cleaned.fasta"

    if extraction_type == "ifeature":
        ifeatures = IfeatureFeatures(ifeature_path)
        extracted_features = ifeatures.extract(fasta_file)

    elif extraction_type == "possum":
        possum = PossumFeatures(
            pssm_dir="pssm",
            output="possum_features",
            program=possum_path,
        )
        extracted_features = possum.extract(fasta_file)

    return extracted_features


def llm_embeddings(
    fasta_file=None,
    model_name="facebook/esm2_t6_8M_UR50D",
    option="mean",
    save_path="embeddings.csv",
    mode="write",
):
    from BioML.deep import embeddings

    embeddings.generate_embeddings(
        fasta_file,
        model_name=model_name,
        option=option,
        save_path=save_path,
        mode=mode,
    )

    return save_path


def feature_selection(
    fasta_file,
    cluster_at_sequence_identity=0.3,
    labels_csv="../data/esterase_labels.csv",
    file_splits=1,
    variance_thres=0.005,
    problem="classification",
    num_threads=2,
    seed=10,
    scaler="robust",
    features_file="classification_results/filtered_features.xlsx",
    extraction_type="ifeature",
    cluster_info="cluster.tsv",
    num_splits=5,
    random_state=100,
    stratified=True,
    num_features_min=20,
    num_features_max=60,
    step_range=10,
):
    import pandas as pd
    from BioML.features import selection
    from BioML.features.extraction import read_features
    from BioML.utilities.split_methods import ClusterSpliter
    from BioML.utilities.utils import MmseqsClustering

    feat = []

    if "ifeature" in extraction_type:
        ifeat = read_features(
            "ifeature", ifeature_out="ifeature_features", file_splits=file_splits
        )
        feat.append(ifeat)
    elif "possum" in extraction_type:
        possum_feat = read_features(
            "possum",
            possum_out="possum_features",
            file_splits=file_splits,
            index=ifeat.index,
        )
        feat.append(possum_feat)
    elif "llm" in extraction_type:
        emb = pd.read_csv("embeddings.csv", index_col=0)
        feat.append(emb)

    features = selection.DataReader(labels_csv, feat, variance_thres=variance_thres)

    #! Why not used the cluster variable?
    # esto genera el "cluster_info"
    _ = MmseqsClustering.easy_cluster(
        fasta_file, cluster_at_sequence_identity=cluster_at_sequence_identity
    )

    split = ClusterSpliter(
        cluster_info,
        num_splits=num_splits,
        random_state=random_state,
        stratified=stratified,
    )
    #! Why not used the y_test variable?
    X_train, X_test, y_train, y_test = split.train_test_split(
        features.features, features.label
    )

    feature_range = selection.get_range_features(
        features.features,
        num_features_min=num_features_min,
        num_features_max=num_features_max,
        step_range=step_range,
    )

    if problem == "classification":
        filters = selection.FeatureClassification()
    elif problem == "regression":
        filters = selection.FeatureRegression()

    select = selection.FeatureSelection(
        features_file,
        filters,
        num_thread=num_threads,
        seed=seed,
        scaler=scaler,
    )

    select.construct_features(
        features.features, X_train, X_test, y_train, feature_range
    )

    return features_file


def outlier_detection(
    features_file="classification_results/filtered_features.xlsx",
    output="classification_results/outliers.csv",
    num_thread=4,
):
    from BioML.utilities.outlier import OutlierDetection

    detection = OutlierDetection(
        features_file,
        output=output,
        num_thread=num_thread,
    )
    outliers = detection.run()

    # outliers dict = outliers
    return output


def model_training(
    problem="classification",
    features="../data/esterase_features.xlsx",
    labels="../data/esterase_labels.csv",
    sheets="ch2_20",
    cluster_info="cluster.tsv",
    num_splits=5,
    random_state=100,
    stratified=True,
    seed=250,
    budget_time=20,
    best_model=3,
    output_path="classification_results",
    optimize="MCC",
    test_size=0.2,
    num_iter=50,
    cross_validation=True,
):
    from BioML.models.base import DataParser, PycaretInterface, Trainer
    from BioML.models.classification import Classifier
    from BioML.models.regression import Regressor
    from BioML.utilities.split_methods import ClusterSpliter
    from BioML.utilities.utils import write_results
    from sklearn.linear_model import PassiveAggressiveClassifier

    data = DataParser(
        features=features,
        label=labels,
        sheets=sheets,
    )
    split = ClusterSpliter(
        cluster_info,
        num_splits=num_splits,
        random_state=random_state,
        stratified=stratified,
    )
    X_train, X_test, _, _ = split.train_test_split(
        data.features, data.features[data.label]
    )

    #! Seems that there are varius models trainings

    experiment = PycaretInterface(
        problem,
        seed,
        budget_time=budget_time,
        best_model=best_model,
        output_path=output_path,
        optimize=optimize,
    )
    if problem == "classification":
        args = Classifier(
            optimize=optimize,
            drop=(),
            selected=(),
            add=(),
            plot=("learning", "confusion_matrix", "class_report"),
        )
    elif problem == "regression":
        args = Regressor(
            optimize=optimize,
            drop=(),
            selected=(),
            add=(),
            plot=("learning", "residuals", "error"),
        )

    training = Trainer(
        experiment,
        args,
        num_splits=num_splits,
        test_size=test_size,
        num_iter=num_iter,
        cross_validation=cross_validation,
    )
    results, models = training.generate_training_results(
        X_train, data.label, tune=True, test_data=X_test, fold_strategy=split
    )

    test_set_predictions = training.generate_holdout_prediction(models)

    training_output = "classification_results"
    l = []
    for tune_status, result_dict in results.items():
        for key, value in result_dict.items():
            write_results(f"{training_output}/{tune_status}", *value, sheet_name=key)
        write_results(
            f"{training_output}/{tune_status}",
            test_set_predictions[tune_status],
            sheet_name=f"test_results",
        )

    from BioML.models import save_model

    generate = save_model.GenerateModel(training)
    for status, model in models.items():
        for key, value in model.items():
            if key == "holdout":
                for num, (name, mod) in enumerate(value.items()):
                    if num > training.experiment.best_model - 1:
                        break
                    final_model = generate.finalize_model(value, num)
                    generate.save_model(
                        final_model,
                        f"classification_results/saved_models/{status}_{name}",
                    )
            else:
                final_model = generate.finalize_model(value)
                generate.save_model(
                    final_model, f"classification_results/saved_models/{key}_{status}"
                )


def prediction(
    training_features="classification_results/filtered_features.xlsx",
    label="../data/esterase_labels.csv",
    outlier_train=(),
    outlier_test=(),
    sheet_name="chi2_60",
    problem="classification",
    model_path="classification_results/saved_models/tuned_rbfsvm",
    scaler="robust",
    fasta="../data/whole_sequence.fasta",
    res_dir="prediction_results_domain",
):

    from BioML.models import predict

    feature = predict.DataParser(
        training_features, label, outliers=outlier_train, sheets=sheet_name
    )
    test_features = feature.remove_outliers(
        feature.read_features(training_features, sheet_name), outlier_test
    )
    predictions = predict.predict(test_features, model_path, problem)

    predictions.index = [f"sample_{x}" for x, _ in enumerate(predictions.index)]
    col_name = ["prediction_score", "prediction_label", "AD_number"]
    predictions = predictions.loc[
        :, predictions.columns.str.contains("|".join(col_name))
    ]

    extractor = predict.FastaExtractor(fasta, res_dir)
    positive, negative = extractor.separate_negative_positive(predictions)
    extractor.extract(
        positive,
        negative,
        positive_fasta="classification_results/positive.fasta",
        negative_fasta=f"classification_results/negative.fasta",
    )


localIPs = {"cactus": "84.88.51.217", "blossom": "84.88.51.250", "bubbles": "84.88.51.219", "phastos": "84.88.187.187"}


def setup_bsc_calculations_based_on_horus_remote(
    remote_name,
    remote_host: str,
    jobs,
    partition,
    scriptName,
    cpus,
    job_name,
    program,
    modulePurge,
    cpus_per_task,
):
    import bsc_calculations

    cluster = "local"
    print("remote_name: ", remote_name)
    print("remote_host: ", remote_host)


    if remote_name != "local":
        cluster = remote_host

    if remote_host in localIPs.values():
        reverse_localIPs = {v: k for k, v in localIPs.items()}
        cluster = reverse_localIPs[remote_host]


    # If we are working with pele, only marenostrum and nord3 are allowed
    if program == "pele":
        if cluster not in [
            "glogin1.bsc.es",
            "glogin2.bsc.es",
            "glogin3.bsc.es",
            "glogin4.bsc.es",
            "nord3.bsc.es",
        ]:
            raise Exception("Pele can only be run on Marenostrum or Nord3")

        elif "glogin" in cluster:
            bsc_calculations.mn5.setUpPELEForMarenostrum(
                jobs,
                partition=partition,
                cpus=cpus,
                general_script=scriptName,
                scripts_folder=scriptName + "_scripts",
            )

        return cluster

    ## Define cluster
    # cte_power
    # if cluster == "plogin1.bsc.es":
    #     bsc_calculations.cte_power.jobArrays(
    #         jobs,
    #         job_name=job_name,
    #         partition=partition,
    #         program=program,
    #         script_name=scriptName,
    #         gpus=cpus,
    #         module_purge=modulePurge,
    #     )
    # marenostrum
    
    elif "glogin" in cluster or "alogin" in cluster:
        print("Generating Marenostrum jobs...")
        bsc_calculations.mn5.singleJob(
            jobs,
            job_name=job_name,
            partition=partition,
            program=program,
            script_name=scriptName,
            ntasks=cpus,
            cpus_per_task=cpus_per_task,
        )

    # cte-amd
    elif "amdlogin" in cluster:
        print("Generating cte-amd jobs...")
        bsc_calculations.amd.jobArrays(
            jobs,
            job_name=job_name,
            partition=partition,
            program=program,
            script_name=scriptName,
            cpus=cpus,
            # module_purge=modulePurge,
        )
    # powerpuff
    elif cluster in ["powerpuff", "phastos", "blossom"]:
        print("Generating powerpuff girls jobs...")
        if cluster == "phastos":
            #jobs = jobs.replace("python", "/home/phastos/Programs/mambaforge/envs/bioml/bin/python")
            jobs = "/home/phastos/Programs/mambaforge/bin/conda run -n bioml " + jobs
        elif cluster == "blossom":
            #jobs = jobs.replace("python", "/home/blossom/Programs/mamba/envs/bioml/bin/python")
            jobs = "/home/blossom/Programs/mamba/condabin/conda run -n bioml " + jobs
            bsc_calculations.local.parallel(
            [f"{jobs}"],
            cpus=min(40, len([jobs])),
            script_name=scriptName,
        )
    # local
    elif cluster == "local":
        print("Generating local jobs...")
        print("Jobs", jobs)
        jobs = "conda run -n bioml " + jobs
        bsc_calculations.local.parallel(
            [f"{jobs}"],
            cpus=min(10, len([jobs])),
            script_name=scriptName,
        )

    else:
        raise Exception("Cluster not supported.")

    return cluster


HOOK_SCRIPT = """
for script in calculation_script.sh_?; do
    sh "$script" > "${script%.*}.out" 2> "${script%.*}.err" &
    exit_code=$?
done

# Wait for all background processes to finish
wait

if [ $exit_code -ne 0 ]; then
    echo "Error: Script $script failed with exit code $exit_code" >&2
    exit 1
fi

# Check if the .err file is empty in order to determine
# if the script ran successfully
if grep -i 'error' "${script%.*}.err" > /dev/null; then
    echo "Error: Script $script failed with the following errors:" >&2
    grep -i 'error' "${script%.*}.err" >&2
    exit 1
fi


echo "All scripts completed successfully."

"""


def launchCalculationAction(
    block: SlurmBlock,
    jobs: str,
    program: str,
    uploadFolders: typing.Optional[typing.List[str]] = None,
    modulePurge: typing.Optional[bool] = False,
):
    if jobs is None:
        raise Exception("No jobs selected")

    partition = block.variables.get("partition")
    cpus = block.variables.get("cpus")
    cpus_per_task = block.variables.get("cpus_per_task")
    simulationName = block.variables.get("folder_name")
    scriptName = block.variables.get("script_name", "calculation_script.sh")

    if simulationName is None:
        simulationName = block.flow.name.lower().replace(" ", "_")

    block.extraData["simulationName"] = simulationName

    print(f"Launching BSC calculation with {cpus} CPUs")
    
    host = "local"
    if hasattr(block.remote, "host"):
        host = block.remote.host

    cluster = setup_bsc_calculations_based_on_horus_remote(
        block.remote.name.lower(),
        host,
        jobs,
        partition,
        scriptName,
        cpus,
        simulationName,
        program,
        modulePurge,
        cpus_per_task,
    )

    # Read the environment variables
    environmentValues = block.variables.get("environment_list", [])
    environmentListValues = {}
    if environmentValues is not None:
        for env in environmentValues:
            environmentListValues[env["environment_key"]] = env["environment_value"]

    # Rewrite the main script to add the environment variables
    # and allow for waiting for the jobs to finish
    # This is only necessary for powerpuff and local
    if cluster in ["phastos", "powerpuff", "local", "blossom"]:
        with open(scriptName, "w") as f:
            f.write("#!/bin/sh\n")

            for key, value in environmentListValues.items():
                f.write(f"export {key}={value}\n")

            f.write(HOOK_SCRIPT)

    if cluster != "local":
        savedID_and_date = block.flow.savedID + "_" + str(datetime.datetime.now().timestamp())
        simRemoteDir = os.path.join(block.remote.workDir, savedID_and_date)
        block.extraData["remoteDir"] = simRemoteDir
        block.remote.remoteCommand(f"mkdir -p -v {simRemoteDir}")

        print(f"Created simulation folder in the remote at {simRemoteDir}")
        print("Sending data to the remote...")

        # Check if in the input, scpefic folders to upload are specified
        # If so, upload them
        if uploadFolders is not None:
            for file in uploadFolders:
                finalPath = block.remote.sendData(file, simRemoteDir)
            block.extraData["uploadedFolder"] = False
        else:
            # Send the whole folder to the remote
            simRemoteDir = block.remote.sendData(os.getcwd(), simRemoteDir)
            block.extraData["uploadedFolder"] = True

        block.extraData["remoteContainer"] = simRemoteDir
        # base_folder = os.path.basename(os.getcwd())

        # # Move the contents of the sent folder to its parent
        # # This is done because the folder is sent as a subfolder
        # command = f"command: mv {simRemoteDir}/{base_folder} {simRemoteDir}"
        # block.remote.remoteCommand(f"mv {simRemoteDir}/{base_folder} {simRemoteDir}")

        # # Remove the sent folder
        # block.remote.remoteCommand(f"rm -rf {simRemoteDir}/{base_folder}")

        # Upload the commands
        for file in os.listdir("."):
            if file.startswith(scriptName):
                block.remote.sendData(file, simRemoteDir)

        # Upload the script
        scriptPath = block.remote.sendData(scriptName, simRemoteDir)

        print("Data sent to the remote.")

        print("Running the simulation...")

        # Run the simulation
        if cluster in ["powerpuff", "phastos", "blossom"]:
            # The powerpuff cluster doesn't have Slurm, so we need to run the script manually & load the Schrodinger module
            command = f"cd {simRemoteDir} && bash {scriptName}"
            block.remote.remoteCommand(command)

        else:
            print(f"Submitting the job to the remote... {scriptPath}")
            if program == "pele":
                with block.remote.cd(simRemoteDir):
                    for jobScript in os.listdir(scriptName + "_scripts"):
                        if jobScript.endswith(".sh"):
                            jobID = block.remote.submitJob(
                                scriptPath + "_scripts/" + jobScript, changeDir=False
                            )
                            print("Submitted job with ID: ", jobID)
                print("Waiting for the jobs to finish...")
            else:

                jobID = block.remote.submitJob(scriptPath)
                print(f"Simulation running with job ID {jobID}. Waiting for it to finish...")

    # * Local
    else:
        print("Running the simulation locally...")

        oldEnv = os.environ.copy()

        for key, value in environmentListValues.items():
            os.environ[key] = value

        # Run the simulation
        try:
            with subprocess.Popen(
                ["sh", scriptName],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ) as p:
                print(f"Simulation running with PID {p.pid}. Waiting for it to finish...")

                if p.stdout is None:
                    raise Exception("No stdout produced by the process")

                for line in p.stdout:
                    strippedOut = line.decode("utf-8").strip()
                    if strippedOut != "":
                        print(strippedOut)

                # Print the error
                strippedErr = ""
                if p.stderr:
                    for line in p.stderr:
                        strippedErr = line.decode("utf-8").strip()
                        if strippedErr != "":
                            print(strippedErr)

                # Wait for the process to finish
                p.wait()

                if p.returncode != 0:
                    raise Exception(strippedErr)
        finally:
            os.environ = oldEnv


def downloadResultsAction(block: SlurmBlock):
    """
    Final action of the block. It downloads the results from the remote.

    Args:
        block (SlurmBlock): The block to run the action on.
    """

    if block.remote.name != "Local":
        cluster = block.remote.host
    else:
        cluster = "local"

    if cluster != "local":
        simRemoteDir = block.extraData["remoteDir"]

        print("Calculation finished, downloading results...")

        currentFolder = os.getcwd()
        folderDestinationOverride = os.path.join(currentFolder, "tmp_download")

        if os.path.exists(folderDestinationOverride):
            shutil.rmtree(folderDestinationOverride)

        # Create the folder
        os.makedirs(folderDestinationOverride)

        final_path = block.remote.getData(simRemoteDir, folderDestinationOverride)

        # If we sent the whole folder, the results are in a subfolder
        # Move them to the parent folder
        if block.extraData.get("uploadedFolder", False):
            print("Uploaded folder, moving results to parent folder")
            final_path = os.path.join(final_path, os.path.basename(currentFolder))

        # Move the contents of the downloaded folder to its parent
        # This is done because the folder is downloaded as a subfolder
        for file in os.listdir(final_path):
            current_path = os.path.join(final_path, file)
            new_path = os.path.join(currentFolder, file)

            if os.path.exists(new_path):
                if os.path.isdir(new_path):
                    shutil.rmtree(new_path)
                else:
                    os.remove(new_path)

            shutil.move(current_path, new_path)

        # Remove the downloaded folder
        shutil.rmtree(folderDestinationOverride)

        final_path = currentFolder

        print(f"Results downloaded to {final_path}")

        remoteContainer = block.extraData["remoteContainer"]

        remove_remote_folder_on_finish = block.variables.get("remove_folder_on_finish", True)
        # Remove the remote folder
        if remove_remote_folder_on_finish:
            print(f"Removing remote folder {remoteContainer}")
            block.remote.remoteCommand(f"rm -rf {remoteContainer}")
    else:
        final_path = os.path.join(os.getcwd())
        print("Calculation finished, results are in the folder: ", final_path)

    return final_path

def load_config(file_path: str, extension: str="json") -> dict:
    """
    Load a configuration file and return its contents as a dictionary.

    Parameters
    ----------
    file_path : str or Path
        The path to the configuration file.
    extension : str, optional
        The file extension. Defaults to "json".

    Returns
    -------
    dict
        The contents of the configuration file as a dictionary.

    Raises
    ------
    ValueError
        If the file extension is not supported.

    Examples
    --------
    >>> load_config("path/to/config.json")
    {'key1': 'value1', 'key2': 'value2'}
    """
    import json
    import yaml
    file_path = Path(file_path)
    if file_path.exists():
        with open(file_path) as file:
            if extension == "json":
                return json.load(file)
            elif extension == "yaml":
                return yaml.load(file, Loader=yaml.FullLoader)
            else:
                raise ValueError(f"Unsupported file extension: {extension}")
    else:
        print("no concent found returning empty {}")
        return {}
    

# Other variables
simulationNameVariable = PluginVariable(
    name="Simulation name",
    id="folder_name",
    description="Name of the simulation folder. By default it will be the same as the flow name.",
    type=VariableTypes.STRING,
    category="Slurm configuration",
)
scriptNameVariable = PluginVariable(
    name="Script name",
    id="script_name",
    description="Name of the script.",
    type=VariableTypes.STRING,
    defaultValue="calculation_script.sh",
    category="Slurm configuration",
)

partitionVariable = PluginVariable(
    name="Partition",
    id="partition",
    description="Partition where to lunch.",
    type=VariableTypes.STRING_LIST,
    defaultValue="gp_bscls",
    allowedValues=["gp_bscls", "gp_debug", "acc_bscls", "acc_debug", "debug", "bsc_ls"],
    category="Slurm configuration",
)

cpusVariable = PluginVariable(
    name="CPUs",
    id="cpus",
    description="Number of CPUs to use.",
    type=VariableTypes.INTEGER,
    defaultValue=1,
    category="Slurm configuration",
)

cpusPerTaskVariable = PluginVariable(
    name="CPUs per task",
    id="cpus_per_task",
    description="Number of CPUs per task to use.",
    type=VariableTypes.INTEGER,
    defaultValue=1,
    category="Slurm configuration",
)

removeFolderOnFinishVariable = PluginVariable(
    name="Remove remote folder on finish",
    id="remove_folder_on_finish",
    description="Deletes the calculation folder on the remote on finish.",
    type=VariableTypes.BOOLEAN,
    defaultValue=True,
    category="Remote",
)

# Advanced variables
environmentKeyVariable = PluginVariable(
    name="Environment",
    id="environment_key",
    description="Environment key",
    type=VariableTypes.STRING,
    category="Environment",
)

environmentValueVariable = PluginVariable(
    name="Value",
    id="environment_value",
    description="Environment value",
    type=VariableTypes.STRING,
    category="Environment",
)

environmentList = VariableList(
    id="environment_list",
    name="Environment variables",
    description="Environment variables to set during the remote connection.",
    prototypes=[environmentKeyVariable, environmentValueVariable],
    category="Environment",
)

BSC_JOB_VARIABLES = [
    simulationNameVariable,
    scriptNameVariable,
    partitionVariable,
    cpusVariable,
    environmentList,
    removeFolderOnFinishVariable,
    cpusPerTaskVariable,
]