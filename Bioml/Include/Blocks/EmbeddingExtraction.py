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

# ==========================#
# Variable outputs
# ==========================#

outputEmbedding = PluginVariable(
    name="output embedding",
    id="out_embedding",
    description="The path to the embedding file in csv or parquet format",
    type=VariableTypes.FILE,
    allowedValues=["csv", "parquet"],
)


outputfile = PluginVariable(
    name="output csv",
    id="output_csv",
    description="The path to the output embeddings in csv format",
    type=VariableTypes.STRING,
    defaultValue="embeddings.csv",
)


##############################
#       Other variables      #
##############################
modelName = PluginVariable(
    name="model name",
    id="model_name",
    description="The name of the LLM model compatible with huggingface",
    type=VariableTypes.STRING,
    defaultValue="facebook/esm2_t33_650M_UR50D",
)

Mode = PluginVariable(
    name="writing mode",
    id="mode",
    description="""Append or write: Whether to write all the embeddings 
                    at once or one batch at the time""",
    type=VariableTypes.STRING_LIST,
    defaultValue="append",
    allowedValues=["append", "write"],
)

batchSize = PluginVariable(
    name="batch size",
    id="batch_size",
    description="The batch size to use.",
    type=VariableTypes.INTEGER,
    defaultValue=8
)

Seed = PluginVariable(
    name="Seed",
    id="seed",
    description="The seed for the random state.",
    type=VariableTypes.INTEGER,
    defaultValue=63462634,
)

Option = PluginVariable(
    name="Option",
    id="option",
    description="The option to concatenate the embeddings.",
    type=VariableTypes.STRING_LIST,
    defaultValue="mean",
    allowedValues=["mean", "sum", "max", "flatten"],
)

Format = PluginVariable(
    name="Format",
    id="format",
    description="The format to use.",
    type=VariableTypes.STRING_LIST,
    defaultValue="csv",
    allowedValues=["csv", "parquet"],
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

pretrainedConfig = PluginVariable(
    name="Pretrained Config",
    id="pretrained_config",
    description="The config file to use for AutoModel.from_pretrained function in json or yaml format",
    type=VariableTypes.FILE,
    defaultValue=None,
    allowedValues=["json", "yaml"],
)

def GenerateEmbeddings(block: SlurmBlock):
    
    input_fasta = block.inputs.get("fasta_file", None)
    if input_fasta is None:
        raise Exception("No input fasta provided")
    if not os.path.exists(input_fasta):
        raise Exception(f"The input fasta file does not exist: {input_fasta}")
    
    if not block.remote.isLocal:
        input_fasta = block.remote.sendData(input_fasta, block.remote.workDir)
    
    output_file = block.variables.get("output_csv", "embeddings.csv")
    model_name = block.variables.get("model_name", "facebook/esm2_t33_650M_UR50D")
    mode = block.variables.get("mode", "append")
    batch_size = block.variables.get("batch_size", 8)
    seed = block.variables.get("seed", 12891245318)
    option = block.variables.get("option", "mean")
    format = block.variables.get("format", "csv")
    llm_config = block.variables.get("llm_config", None)
    tokenizer_config = block.variables.get("tokenizer_config", None)
    pretrained_config = block.variables.get("pretrained_config", None)
    
    if llm_config and not block.remote.isLocal:
        llm_config = block.remote.sendData(llm_config, block.remote.workDir)
    if tokenizer_config and not block.remote.isLocal:
        tokenizer_config = block.remote.sendData(tokenizer_config, block.remote.workDir)
    if pretrained_config and not block.remote.isLocal:
        pretrained_config = block.remote.sendData(pretrained_config, block.remote.workDir)
    
        
    ## extradata
    block.extraData["output_embedding"] = output_file
    block.extraData["format"] = format
    
    command = f"python -m BioML.deep.embeddings "
    command += f"{input_fasta} "
    command += f"-m {model_name} "
    command += f"-d {mode} "
    command += f"-p {output_file} "
    command += f"-b {batch_size} "
    command += f"-s {seed} "
    command += f"-f {format} "
    command += f"-op {option} "
    if llm_config:
        command += f"-l {llm_config} "
    if tokenizer_config:
        command += f"-t {tokenizer_config} "
    if pretrained_config:
        command += f"-pt {pretrained_config} "

    jobs = command
    
    ## change bsc variables
    block.variables["script_name"] = "generate_embeddings.sh"
    block.variables["partition"] = "acc_bscls"

    from utils import launchCalculationAction
    
    launchCalculationAction(
        block,
        jobs,
        program="bioml",
        uploadFolders=[]
    )


def finalAction(block: SlurmBlock):
    from pathlib import Path
    from utils import downloadResultsAction

    downloaded_path = downloadResultsAction(block)
    output_embedding = block.extraData["output_embedding"]
    format = block.extraData["format"]
    
    if format == "csv":
        block.setOutput(outputEmbedding.id, str(Path(downloaded_path)/output_embedding))
    else:
        block.setOutput(outputEmbedding.id, str((Path(downloaded_path)/output_embedding).with_suffix(".parquet")))
    


from utils import BSC_JOB_VARIABLES

EmbeddingBlock = SlurmBlock(
    name="Embeddings",
    id="embeddings",
    initialAction=GenerateEmbeddings,
    finalAction=finalAction,
    description="Generate LLM embeddings",
    inputs=[fastaFile],
    variables=BSC_JOB_VARIABLES
    + [
        modelName,
        Mode,
        batchSize,  
        outputfile,
        Seed,
        Option,
        Format,
        LLMConfig,
        TokenizerConfig,
        pretrainedConfig
    ],
    outputs=[outputEmbedding],
)

