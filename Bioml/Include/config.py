from HorusAPI import (
    PluginVariable,
    SlurmBlock,
    VariableGroup,
    VariableList,
    VariableTypes,
    PluginConfig
)

possumProgram = PluginVariable(
    name="possum program",
    id="possum_dir",
    description="The path to the possum program directory, it should be the remote path",
    type=VariableTypes.STRING,
    defaultValue="POSSUM_Toolkit/",
)

databaseInput = PluginVariable(
    name="database input",
    id="dbin",
    description="""The path to database fasta file, it should already be in remote, so use the remote path, 
    even if it is a fasta file you want to create the database from""",
    type=VariableTypes.STRING,
    defaultValue=None,
)

ifeatureDir = PluginVariable(
    name="Ifeature dir",
    id="ifeature_dir",
    description="The path to iFeature program",
    type=VariableTypes.STRING,
    defaultValue="iFeature",
)

BIOML_CONFIG = PluginConfig(
    name="BIOML",
    id="bioml_config",
    description="The configuration for the BIOML pipeline.",
    variables=[
        possumProgram,
        # Add other variables here
        databaseInput,
        ifeatureDir
    ],
)

