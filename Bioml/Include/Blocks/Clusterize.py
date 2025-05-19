"""
A module that clusters sequences using Mmseqs2
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

clusterAtSeqIdentity = PluginVariable(
    name="Cluster at seq identity",
    id="cluster_at_seq_identity",
    description="The sequence identity to use for clustering",
    type=VariableTypes.FLOAT,
    defaultValue=0.3,
)

# ==========================#
# Variable outputs
# ==========================#
outputTsv = PluginVariable(
    name="cluster file",
    id="cluster_file",
    description="The output file for cluster",
    type=VariableTypes.STRING,
    defaultValue="cluster.tsv",
)

outputCluster = PluginVariable(
    name="output cluster",
    id="out_cluster",
    description="The path to the cluster file",
    type=VariableTypes.FILE,
)




def runClusterizeBioml(block: SlurmBlock):
    
    from pathlib import Path 
    
    input_fasta = block.inputs.get("fasta_file", None)
    if input_fasta is None:
        raise Exception("No input fasta provided")
    if not os.path.exists(input_fasta):
        raise Exception(f"The input fasta file does not exist: {input_fasta}")
    
    input_fasta = block.remote.sendData(input_fasta, block.remote.workDir)
    cluster_at_seq_identity = block.variables.get("cluster_at_seq_identity", 0.3)

    ## variables
    output_tsv = block.variables.get("cluster_file", "cluster.tsv")
    
    ## extradata
    block.extraData["cluster_file"] = output_tsv

    command = f"python -m BioML.features.generate_pssm "
    command += f"-i {input_fasta} "
    command += f"-cluster_at_sequence_identity {cluster_at_seq_identity} "
    command += f"--cluster_file {output_tsv} "
    command += f"-c  "

    jobs = command
    

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

    out_cluster = block.extraData["cluster_file"]
    
    block.setOutput(outputCluster.id, str(Path(downloaded_path)/out_cluster))



from utils import BSC_JOB_VARIABLES

ClusterizeBlock = SlurmBlock(
    name="Clusterize",
    id="clusterize",
    initialAction=runClusterizeBioml,
    finalAction=finalAction,
    description="Clusterize sequences using Mmseqs2",
    inputs=[fastaFile],
    variables=BSC_JOB_VARIABLES
    + [
        outputTsv,
        clusterAtSeqIdentity
    ],
    outputs=[outputCluster],
)

