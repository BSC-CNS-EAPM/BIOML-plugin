"""
Bioml Classification
    | Wrapper class for the bioml Classification module.
    | Train classification models.
"""

from HorusAPI import PluginVariable, SlurmBlock, VariableTypes

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
    defaultValue="extract",
)

# ================================#
# Variable outputs for extraction
# ================================#
outputExtraction = PluginVariable(
    name="Feature Extraction output",
    id="out_zip",
    description="The features extracted",
    type=VariableTypes.FILE,
)

outputIfeature = PluginVariable(
    name="Feature Extraction output",
    id="out_zip",
    description="The features extracted",
    type=VariableTypes.FILE,
)

outputPossum = PluginVariable(
    name="Feature Extraction output",
    id="out_zip",
    description="The features extracted",
    type=VariableTypes.FILE,
)


# ================================#
# Variables for filtering
# ================================#

Sheets = PluginVariable(
    name="sheets",
    id="sheets",
    description="Excel sheet name (from theselected features) to use for filtering the new features",
    type=VariableTypes.STRING,
    defaultValue=None,
)

inputFeatures = PluginVariable(
    name="Input features",
    id="training_features",
    description="The path to the training features in excel format used for filtering data",
    type=VariableTypes.FILE,
)

outputNewFeatures = PluginVariable(
    name="Output new features",
    id="new_features",
    description="The path to the new features in CSV format",
    type=VariableTypes.FILE,
)

##############################
#       Other variables      #
##############################

inputPSSM = PluginVariable(
    name="PSSM dir input",
    id="pssm_dir",
    description="The path to the folder with the PSSM files",
    type=VariableTypes.FOLDER,
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




def initialAction(block: SlurmBlock):
    block.variables["script_name"] = "generate_pssm.sh"

    clean_fasta(
        "/home/ruite/Projects/enzyminer/POSSUM_Toolkit",
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
            program="/home/ruite/Projects/enzyminer/POSSUM_Toolkit",
        )
        extracted_features = possum.extract(fasta_file)

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
