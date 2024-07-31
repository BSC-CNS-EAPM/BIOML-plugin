"""
Bioml Model training
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
inputOutliers = PluginVariable(
    name="Input Outliers",
    id="out_csv",
    description="The CSV file with the input data",
    type=VariableTypes.FILE,
    allowedValues=["csv"],
)

# ==========================#
# Variable outputs
# ==========================#
outputtraining = PluginVariable(
    name="Models output",
    id="out_zip",
    description="The features extracted",
    type=VariableTypes.FILE,
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
    inputs=[inputCsv, inputOutliers],
    variables=BSC_JOB_VARIABLES + [],
    outputs=[outputtraining],
)
