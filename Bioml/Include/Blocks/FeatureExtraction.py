"""
Bioml Classification
    | Wrapper class for the bioml Classification module.
    | Train classification models.
"""

from HorusAPI import PluginVariable, SlurmBlock, VariableTypes

# ==========================#
# Variable inputs
# ==========================#
inputCsv = PluginVariable(
    name="Input CSV",
    id="in_csv",
    description="The CSV file with the input data",
    type=VariableTypes.FILE,
    allowedValues=["csv"],
)

# ==========================#
# Variable outputs
# ==========================#
outputExtraction = PluginVariable(
    name="Feature Extraction output",
    id="out_zip",
    description="The features extracted",
    type=VariableTypes.FILE,
)


def initialAction(block: SlurmBlock):
    from BioML.features.extraction import (
        IfeatureFeatures,
        PossumFeatures,
        read_features,
    )
    from BioML.utilities.utils import clean_fasta

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
