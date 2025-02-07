"""
Entry point for the Bioml plugin
"""

from HorusAPI import Plugin


def createPlugin():
    """
    Generates the Bioml plugin and returns the instance
    """
    # ========== Plugin Definition ========== 

    biomlPlugin = Plugin(id="bioml")

    # ========== Blocks ========== #
    from Blocks.Regression import regressionBlock  # type: ignore

    biomlPlugin.addBlock(regressionBlock)

    from Blocks.Classification import classificationBlock  # type: ignore

    biomlPlugin.addBlock(classificationBlock)

    from Blocks.FeatureExtraction import featureExtractionBlock  # type: ignore

    biomlPlugin.addBlock(featureExtractionBlock)

    from Blocks.FeatureSelection import featureSelectionBlock  # type: ignore

    biomlPlugin.addBlock(featureSelectionBlock)

    from Blocks.ModelTraining import SaveModelBlock  # type: ignore

    biomlPlugin.addBlock(SaveModelBlock)

    from Blocks.Outliers import outliersBlock  # type: ignore

    biomlPlugin.addBlock(outliersBlock)
    
    from Blocks.PssmGeneration import generatePSSMBlock  # type: ignore

    biomlPlugin.addBlock(generatePSSMBlock)

    from Blocks.Prediction import PredictBlock  # type: ignore

    biomlPlugin.addBlock(PredictBlock)
    
    from Blocks.EmbeddingExtraction import EmbeddingBlock  # type: ignore

    biomlPlugin.addBlock(EmbeddingBlock)

    # Return the plugin
    return biomlPlugin


plugin = createPlugin()
