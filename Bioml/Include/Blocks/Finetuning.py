"""
A module that extracts LMM embeddings
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
# Variable inputs
# ==========================#
fastaFile = PluginVariable(
    name="fasta file",
    id="fasta_file",
    description="The fasta file path.",
    type=VariableTypes.FILE,
    defaultValue=None,
    allowedValues=["fasta", "fsa"],
)

inputLabelFile = PluginVariable(
    name="Input Label File",
    id="input_label_file",
    description="The path to the labels of the training set in a csv file, a file compatible with np.load",
    type=VariableTypes.FILE,
    defaultValue=None,
)

# ==========================#
# Variable outputs
# ==========================#

outputPEFT = PluginVariable(
    name="output peft adapter",
    id="out_peft",
    description="The path to the adapter_model models",
    type=VariableTypes.FOLDER,
)


##############################
#       Other variables      #
##############################
learningRate= PluginVariable(
    name="learning rate",
    id="learning_rate",
    description="The learning rate to use.",
    type=VariableTypes.FLOAT,
    defaultValue=1e-3,
)

UseBestModel = PluginVariable(
    name="Use Best Model",
    id="use_best_model",
    description="If to use the best model saved in the checkpoint",
    type=VariableTypes.BOOLEAN,
    defaultValue=True,
)

LoraInit = PluginVariable(
    name="Lora Init",
    id="lora_init",
    description="How to initialize lora weights",
    type=VariableTypes.STRING,
    defaultValue="True",
)

LLMConfig = PluginVariable(
    name="LLM Config",
    id="llm_config",
    description="The config file to use for LLM model in json or yaml format",
    type=VariableTypes.FILE,
    defaultValue=None,
    allowedValues=["json", "yaml"],
)

LightningConfig = PluginVariable(
    name="Lightning Config",
    id="lightning_config",
    description="The config file to use for the lightninh trainer in json or yaml format",
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

splitConfig = PluginVariable(
    name="Split Config",
    id="split_config",
    description="The config file to use for splitting data in json or yaml format",
    type=VariableTypes.FILE,
    defaultValue=None,
    allowedValues=["json", "yaml"],
)

trainConfig = PluginVariable(
    name="Train Config",
    id="train_config",
    description="The config file to use for training in json or yaml format",
    type=VariableTypes.FILE,
    defaultValue=None,
    allowedValues=["json", "yaml"],
)

def Finetune(block: SlurmBlock):
    
    from pathlib import Path
    from utils import load_config
    # inputs
    input_fasta = block.inputs.get("fasta_file", None)
    if input_fasta is None:
        raise Exception("No input fasta provided")
    if not os.path.exists(input_fasta):
        raise Exception(f"The input fasta file does not exist: {input_fasta}")
    
    input_label = block.inputs.get("input_label_file", None)
    if input_label is None:
        raise Exception("No input fasta provided")
    if not os.path.exists(input_label):
        raise Exception(f"The input fasta file does not exist: {input_label}")
    
    # other variables
    llm_config = block.variables.get("llm_config", None)
    tokenizer_config = block.variables.get("tokenizer_config", None)
    split_config = block.variables.get("split_config", None)
    train_config = block.variables.get("train_config", None)
    lightning_config = block.variables.get("lightning_config", None)
    lora_init = block.variables.get("lora_init", "True")
    learning_rate = block.variables.get("learning_rate", 1e-3)
    use_best_model = block.variables.get("use_best_model", True)

    # Create an copy the inputs
    folderName = "finetuning_inputs"
    os.makedirs(folderName, exist_ok=True)
    os.system(f"cp {input_fasta} {folderName}")
    os.system(f"cp {input_label} {folderName}")
    
    if llm_config and not os.path.exists(llm_config):
            raise Exception(f"The LLM config file does not exist: {llm_config}")
    os.system(f"cp {llm_config} {folderName}")

    if tokenizer_config and not os.path.exists(tokenizer_config):
            raise Exception(f"The Tokenizer config file does not exist: {tokenizer_config}")
    os.system(f"cp {tokenizer_config} {folderName}")

    if split_config and  not os.path.exists(split_config):
            raise Exception(f"The Split config file does not exist: {split_config}")
    os.system(f"cp {split_config} {folderName}")

    if train_config and not os.path.exists(train_config):
            raise Exception(f"The Train config file does not exist: {train_config}")
    os.system(f"cp {train_config} {folderName}")

    if lightning_config and not os.path.exists(lightning_config):
            raise Exception(f"The Lightning config file does not exist: {lightning_config}")
    os.system(f"cp {lightning_config} {folderName}")
    
    ## Commands
    command = f"python -m BioML.deep.finetuning "
    command += f"-i {folderName}/{Path(input_fasta).name} "
    command += f"--label {folderName}/{Path(input_label).name} "
    command += f"--lr {learning_rate} "
    command += f"--lora_init {lora_init} "
    if not use_best_model:
        command += "-u "
    if llm_config:
        command += f"-lc {folderName}/{Path(llm_config).name} "
    if tokenizer_config:
        command += f"-tc {folderName}/{Path(tokenizer_config).name} "
    if lightning_config:
        command += f"-lt {folderName}/{Path(lightning_config).name} "
    if split_config:
        command += f"-sp {folderName}/{Path(split_config).name} "
    if train_config:
        command += f"-tr {folderName}/{Path(train_config).name} "


    jobs = [command]
    
    ## change bsc variables
    block.variables["script_name"] = "finetuning.sh"
    block.variables["partition"] = "acc_bscls"
    block.extraData["train_config"] = train_config

    from utils import launchCalculationAction
    
    launchCalculationAction(
        block,
        jobs,
        program="bioml",
        uploadFolders=[folderName]
    )


def finalAction(block: SlurmBlock):
    from pathlib import Path
    from utils import load_config
    from utils import downloadResultsAction
    
    downloaded_path = downloadResultsAction(block)
    train_config = block.extraData.get("train_config", None)
    e = Extensions()

    if train_config is None:
        peft_path = "peft_model"
        csv_file = Path(f"{downloaded_path}/Loggers").glob("*/*/*.csv")
    else:
        train_config = load_config(train_config)
        peft_path = Path(train_config["root_dir"]) / train_config["adapter_output"]
        csv_file = Path(f"{downloaded_path}/{train_config['root_dir']}/Loggers").glob("*/*/metrics.csv")

    for file in csv_file:
         # visualize the performance metrics per epoch of the trained models
         e.loadCSV(str(file), f"{file.parent.name}_metrics")
         
    block.setOutput(
        outputPEFT.id,
        os.path.join(downloaded_path, peft_path),
    )


from utils import BSC_JOB_VARIABLES

FinetuningBlock = SlurmBlock(
    name="Finetuning",
    id="finetuning",
    initialAction=Finetune,
    finalAction=finalAction,
    description="Finetune a LLM model",
    inputs=[fastaFile, inputLabelFile],
    variables=BSC_JOB_VARIABLES
    + [
        learningRate,
        UseBestModel,
        LoraInit,
        LLMConfig,
        TokenizerConfig,
        splitConfig,
        trainConfig,
    ],
    outputs=[outputPEFT],
)

