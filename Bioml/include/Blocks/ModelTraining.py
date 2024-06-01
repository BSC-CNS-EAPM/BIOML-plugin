"""
Bioml Model training
    | Wrapper class for the bioml Classification module.
    | Train classification models.
"""

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

from HorusAPI import (
    PluginVariable,
    SlurmBlock,
    VariableGroup,
    VariableList,
    VariableTypes,
)


def initialAction(block: SlurmBlock):
    pass


def finalAction(block: SlurmBlock):
    pass


from utils import BSC_JOB_VARIABLES

modelTrainingBlock = SlurmBlock(
    name="Module training BioML",
    initialAction=initialAction,
    finalAction=finalAction,
    description="Module training.",
    inputs=[inputCsv],
    variables=BSC_JOB_VARIABLES + [],
    outputs=[outputClassification],
)
