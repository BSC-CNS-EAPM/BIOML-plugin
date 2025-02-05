"""
Bioml Classification
    | Wrapper class for the bioml Classification module.
    | Train classification models.
"""

from HorusAPI import PluginVariable, SlurmBlock, VariableTypes, PluginBlock

# ==============================#
# Variable inputs for extraction
# ==============================#

inputFasta = PluginVariable(
    name="input fasta",
    id="fasta_file",
    description="The path to the fasta file",
    type=VariableTypes.FILE,
)

Purpose = PluginVariable(
    name="purpose",
    id="purpose",
    description="The purpose of the program",
    type=VariableTypes.CHECKBOX,
    choices=["extract", "read", "filter"],
    defaultValue=["extract", "read"],
)

# ================================#
# Variable outputs for extraction
# ================================#
outputExtraction = PluginVariable(
    name="Extracted directory",
    id="extracted_out",
    description="The directory for the extracted features",
    type=VariableTypes.STRING,
    defaultValue="extracted_features",
)

outputIfeature = PluginVariable(
    name="Ifeature output",
    id="ifeature_out",
    description="The folder to save the ifeatures features",
    type=VariableTypes.STRING,
)

outputPossum = PluginVariable(
    name="Feature Extraction output",
    id="possum_out",
    description="The folder to save the possum features",
    type=VariableTypes.STRING,
)

##############################
#       Other variables      #
##############################

inputPSSM = PluginVariable(
    name="PSSM dir input",
    id="pssm_dir",
    description="The path to the folder with the PSSM files",
    type=VariableTypes.STRING,
)


ifeatureDir = PluginVariable(
    name="Ifeature dir",
    id="ifeature_dir",
    description="The path to iFeature program",
    type=VariableTypes.STRING,
    defaultValue="iFeature",
)

possumProgram = PluginVariable(
    name="possum program",
    id="possum_dir",
    description="The path to the possum program directory",
    type=VariableTypes.STRING,
    defaultValue="POSSUM_Toolkit/",
)

LongCommand = PluginVariable(
    name="Long commands",
    id="long_command",
    description="If restarting the programs using only those that takes longer times (because the others have already finished)",
    type=VariableTypes.BOOL,
    defaultValue=False,
)

Run = PluginVariable(
    name="run",
    id="run",
    description="The program to run",
    type=VariableTypes.CHECKBOX,
    allowedValues=["possum", "ifeature"],
    defaultValue=["possum", "ifeature"],
)

numThreads = PluginVariable(
    name="Number of threads",
    id="num_threads",
    description="The number of threads to use.",
    type=VariableTypes.INTEGER,
    defaultValue=100,
)

dropIFeature = PluginVariable(
    name="drop ifeature",
    id="drop_ifeature",
    description="Features from Ifeatures to skip during feature extraction",
    type=VariableTypes.CHECKBOX,
    allowedValues=["APAAC", "PAAC", "CKSAAGP", "Moran", 
                   "Geary", "NMBroto", "CTDC", "CTDT", 
                   "CTDD", "CTriad", "GDPC", "GTPC",
                   "QSOrder", "SOCNumber", "GAAC", 
                   "KSCTriad"],
    defaultValue=None,
)

dropPossum = PluginVariable(
    name="drop possum",
    id="drop_possum",
    description="Features from Ifeatures to skip during feature extraction",
    type=VariableTypes.CHECKBOX,
    allowedValues=["aac_pssm", "ab_pssm", "d_fpssm", "dp_pssm",
                    "dpc_pssm", "edp", "eedp", "rpm_pssm", 
                    "k_separated_bigrams_pssm",
                    "pssm_ac", "pssm_cc",
                    "pssm_composition", "rpssm", 
                    "s_fpssm", "smoothed_pssm:5", 
                    "smoothed_pssm:7", 
                    "smoothed_pssm:9", "tpc", 
                    "tri_gram_pssm", 
                    "pse_pssm:1", "pse_pssm:2", 
                    "pse_pssm:3"],
    defaultValue=None,
)

cleanFasta = PluginVariable(
    name="Clean fasta",
    id="clean_fasta",
    description="The file name to the cleaned fasta file",
    type=VariableTypes.STRING,
    defaultValue=None,
)


##############################
#       Omega variables      #
##############################

OmegaType = PluginVariable(
    name="omega type",
    id="omega_type",
    description="The type of molecule to extract features from using Omega Features",
    type=VariableTypes.STRING_LIST,
   allowedValues=["structure", "RNA", "DNA", "ligand"],
   defaultValue=None
)



def initialAction(block: SlurmBlock):
    ## inputs
    input_fasta = block.inputs.get("fasta_file", None)
    input_pssm = block.variables.get("pssm_dir", None)
    purpose = block.variables.get("purpose", ["extract", "read"])
    ## output folders
    outputposum = block.variables.get("possum_out", "possum_features")
    outputifeature = block.variables.get("ifeature_out", "ifeature_features")
    extracted_features = block.variables.get("extracted_out", "extracted_features")
    clean_fasta = block.variables.get("clean_fasta", None)
    
    ## other variables
    num_threads = block.variables.get("num_threads", 100)
    run = block.variables.get("run", ["possum", "ifeature"])
    drop_ifeature = block.variables.get("drop_ifeature", [])
    drop_possum = block.variables.get("drop_possum", [])
    long = block.variables.get("long_command", False)
    possum_program = block.variables.get("possum_dir", "POSSUM_Toolkit/")
    ifeature_program = block.variables.get("ifeature_dir", "iFeature")
    omega_type = block.variables.get("omega_type", None)
    

    if input_fasta is None:
        raise Exception("No input fasta provided")
    if "possum" in run and input_pssm is None:
        raise Exception("No input pssm folder provided to run the Possum program")

    block.variables["cpus"] = num_threads
    block.variables["script_name"] = "feature_extraction.sh"

    command = f"python -m BioML.features.extract "
    command += f"-i {input_fasta} "
    command += f"-r {' '.join(run)} "

    if input_pssm is not None:
        command += f"-p {input_pssm} "
    command += f"-Po {possum_program} "
    command += f"-id {ifeature_program} "
    command += f"-po {outputposum} "
    command += f"-io {outputifeature} "
    command += f"-on {' '.join(purpose)} "
    if long:
        command += f"--long "
    if omega_type is not None:
        command += f"--omega_type {omega_type} "
    if drop_ifeature + drop_possum:
        command += f"--drop {' '.join(drop_ifeature + drop_possum)} "
    if clean_fasta:
        command += f"-cl {clean_fasta} "

    jobs = [command]


    return extracted_features


def finalAction(block: SlurmBlock):
    pass


from utils import BSC_JOB_VARIABLES

featureExtractionBlock = SlurmBlock(
    name="Feature Extraction BioML",
    initialAction=initialAction,
    finalAction=finalAction,
    description="Feature Extraction.",
    inputs=[inputCsv],
    variables=BSC_JOB_VARIABLES + [],
    outputs=[outputExtraction],
)


ReadFeaturesBlock = PluginBlock(