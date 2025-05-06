"""
A module that performs suggests mutations to a given sequence.
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

### ==========================#
# Variable outputs
# ==========================#
outputFile = PluginVariable(
    name="Output file",
    id="out_file",
    description="The path to the suggestions in csv format",
    type=VariableTypes.STRING,
    defaultValue="suggestions.csv",
)

outputSuggest = PluginVariable(
    name="output suggestions",
    id="out_suggest",
    description="The path to the suggestions file in csv format",
    type=VariableTypes.FILE,
)


### ==========================#
# other Variables
# ==========================#

modelName = PluginVariable(
    name="model name",
    id="model_name",
    description="The name of the LLM model compatible with huggingface",
    type=VariableTypes.STRING,
    defaultValue="facebook/esm2_t33_650M_UR50D",
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


plotVar = PluginVariable(
    name="Plot",
    id="plot",
    description="If to plot the suggestions",
    type=VariableTypes.BOOLEAN,
    defaultValue=True,
)

strategyVar = PluginVariable(
    name="Strategy",
    id="strategy",
    description="The strategy to use for the suggestions.",
    type=VariableTypes.STRING_LIST,
    defaultValue="masked_marginal",
    allowedValues=["masked_marginal", "wild_marginal"],
)

positionVar = PluginVariable(
    name="Position",
    id="position",
    description="The position of the mutation.",
    type=VariableTypes.LIST,
    defaultValue=(),
)

### ==========================#

def GenerateSuggestions(block: SlurmBlock):
    """
    Generates suggestions for a given sequence.
    """

    input_fasta = block.inputs.get("fasta_file", None)
    if input_fasta is None:
        raise Exception("No input fasta provided")
    if not os.path.exists(input_fasta):
        raise Exception(f"The input fasta file does not exist: {input_fasta}")

    input_fasta = block.remote.sendData(input_fasta, block.remote.workDir)

    output_file = block.variables.get("out_file", "suggestions.csv")
    model_name = block.variables.get("model_name", "facebook/esm2_t33_650M_UR50D")
    plot = block.variables.get("plot", True)
    strategy = block.variables.get("strategy", "masked_marginal")
    llm_config = block.variables.get("llm_config", None)
    tokenizer_config = block.variables.get("tokenizer_config", None)
    positions = block.variables.get("position", ())
    
    if llm_config:
        llm_config = block.remote.sendData(llm_config, block.remote.workDir)
    if tokenizer_config:
        tokenizer_config = block.remote.sendData(tokenizer_config, block.remote.workDir)
    
        ## extradata
    block.extraData["output_suggestions"] = output_file
    block.extraData["plot"] = plot

    command = f"python -m BioML.applications.suggest "
    command += f"--fasta {input_fasta} --save_path {output_file} --model_name {model_name} --strategy {strategy}"
    if not plot:
        command += " --plot"
    if llm_config:
        command += f" --llm_config {llm_config}"
    if tokenizer_config:
        command += f" --tokenizer_config {tokenizer_config}"
    if positions:
        command += f" --positions {' '.join(map(str, positions))}"

    jobs = command
    
    ## change bsc variables
    block.variables["script_name"] = "suggest.sh"
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
    
    e = Extensions()

    downloaded_path = downloadResultsAction(block)
    output_suggestions = block.extraData["output_suggestions"]
    plot = block.extraData["plot"]
    
    block.setOutput(outputSuggest.id, str(Path(downloaded_path)/output_suggestions))

    e.loadCSV(
        str(Path(downloaded_path)/output_suggestions),
        "suggestions",
    )

    if plot:
        from pathlib import Path
        path = ((Path(downloaded_path)/output_suggestions).parent / "heatmap").glob("*.png")
        for p in path:
            e.loadImage(str(p), p.stem)


from utils import BSC_JOB_VARIABLES

SuggestionBlock = SlurmBlock(
    name="Suggest mutations",
    description="Suggest mutations to a given sequence.",
    id="suggest_mutations",
    initialAction=GenerateSuggestions,
    finalAction=finalAction,
    inputs=[fastaFile],
    outputs=[outputSuggest],
    variables= BSC_JOB_VARIABLES +[
        modelName,
        LLMConfig,
        TokenizerConfig,
        plotVar,
        strategyVar,
        positionVar,
        outputFile
    ],

)