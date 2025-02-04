"""
A module that generates PSSM files using Mmseqs2
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
    allowedValues=[".fasta", ".fsa"],
)

possumProgram = PluginVariable(
    name="possum program",
    id="possum_dir",
    description="The path to the possum program directory",
    type=VariableTypes.STRING,
    defaultValue="POSSUM_Toolkit/",
)

databaseInput = PluginVariable(
    name="database input",
    id="dbin",
    description="The path to database fasta file",
    type=VariableTypes.STRING,
    defaultValue=None,
)
# ==========================#
# Variable outputs
# ==========================#
outputPSSM = PluginVariable(
    name="PSSM output",
    id="pssm_out",
    description="The path to the output for the outliers.",
    type=VariableTypes.FOLDER,
)

outputFasta = PluginVariable(
    name="output fasta",
    id="out_fasta",
    description="The path to the fixed fasta file",
    type=VariableTypes.FILE,
)


outputDir = PluginVariable(
    name="output directory",
    id="output_dir",
    description="The path to the output for the pssm files",
    type=VariableTypes.STRING,
    defaultValue="pssm",
)

##############################
#       Other variables      #
##############################
numThreads = PluginVariable(
    name="Number of threads",
    id="num_threads",
    description="The number of threads to use.",
    type=VariableTypes.INTEGER,
    defaultValue=100,
)

iteraTions = PluginVariable(
    name="iterations",
    id="iterations",
    description="The number of iterations at running mmseqs2.",
    type=VariableTypes.INTEGER,
    defaultValue=3,
)

eValue = PluginVariable(
    name="e-value",
    id="evalue",
    description="The e-value to use.",
    type=VariableTypes.FLOAT,
    defaultValue=0.01,
)

Sensitivity = PluginVariable(
    name="Sensitivity",
    id="sensitivity",
    description="The sensitivity to use, the higher the more accurate the homology searches",
    type=VariableTypes.FLOAT,
    defaultValue=6.5,
)

generateSearchDB = PluginVariable(
    name="Generate search DB",
    id="generate_search_db",
    description="Generate a database for searching.",
    type=VariableTypes.BOOLEAN,
    defaultValue=False,
)


def runGeneratePSSMBioml(block: SlurmBlock):
    from pathlib import Path 
    
    input_fasta = block.inputs.get("fasta_file", None)
    if input_fasta is None:
        raise Exception("No input fasta provided")
    if not os.path.exists(input_fasta):
        raise Exception(f"The input fasta file does not exist: {input_fasta}")
    
    possum_program_dir = block.inputs.get("possum_dir", None)
    database_input = block.inputs.get("database_input", None)
    if database_input is None:
        raise Exception("No input database provided")
   
    ## variables
    num_threads = block.variables.get("num_threads", 100)
    num_iterations = block.variables.get("iterations", 3)
    e_value = block.variables.get("evalue", 0.01)
    sensitivity = block.variables.get("sensitivity", 6.5)
    generate_search_db = block.variables.get("generate_search_db", False)
    output_dir = block.variables.get("output_dir", "pssm")
    
    ## extradata
    block.extraData["pssm_dir"] = output_dir
    block.extraData["fasta_file"] = str(Path(input_fasta).with_stem(Path(input_fasta).stem + "_fixed"))
    
    ## change bsc variables
    if num_threads > 1:
        block.variables["cpus"] = num_threads
    
    command = f"python -m BioML.features.generate_pssm "
    command += f"-i {input_fasta} "
    command += f"-Po {possum_program_dir} "
    command += f"-dbinp {database_input} "
    command += f"--iterations {num_iterations} "
    command += f"--evalue {e_value} "
    command += f"--sensitivity {sensitivity} "
    command += f"--generate_searchdb {generate_search_db} "
    command += f"-p {output_dir} "

    jobs = [command]
    
    ## change bsc variables
    block.variables["cpus"] = (num_threads // 10) + 1
    block.variables["cpus_per_task"] = 10

    from utils import launchCalculationAction
    
    launchCalculationAction(
        block,
        jobs,
        program="bioml",
        uploadFolders=[
            input_fasta,
        ],
    )


def finalAction(block: SlurmBlock):
    from pathlib import Path
    from utils import downloadResultsAction
    downloaded_path = downloadResultsAction(block)
    out_pssm = block.extraData["pssm_dir"]
    out_fasta = block.extraData["fasta_file"]
    
    block.setOutput(outputFasta.id, Path(downloaded_path)/out_fasta)
    
    block.setOutput(outputPSSM.id, Path(downloaded_path)/out_pssm)


from utils import BSC_JOB_VARIABLES

generatePSSMBlock = SlurmBlock(
    name="Outlier BioMl",
    initialAction=runGeneratePSSMBioml,
    finalAction=finalAction,
    description="Outlier detection.",
    inputs=[fastaFile, possumProgram, databaseInput],
    variables=BSC_JOB_VARIABLES
    + [numThreads,
        iteraTions,
        eValue,
        Sensitivity,
        generateSearchDB,
        outputDir
    ],
    outputs=[outputPSSM, outputFasta],
)

