"""
Entry point for the Bioml plugin
"""

from HorusAPI import Plugin


def createPlugin():
    """
    Generates the Bioml plugin and returns the instance
    """
    # ========== Plugin Definition ========== #

    biomlPlugin = Plugin(id="Bioml")

    # ========== Blocks ========== #
    from Blocks.Classification import classificationBlock  # type: ignore

    biomlPlugin.addBlock(classificationLBlock)

    from Blocks.Outliers import outliersBlock  # type: ignore

    biomlPlugin.addBlock(outliersBlock)

    from Blocks.Prediction import PredictBlock  # type: ignore

    biomlPlugin.addBlock(PredictBlock)

    from Blocks.Regression import regressionBlock  # type: ignore

    biomlPlugin.addBlock(regressionBlock)

    from Blocks.Classification import classificationBlock  # type: ignore

    biomlPlugin.addBlock(classificationBlock)

    from Blocks.FeatureExtraction import featureExtractionBlock  # type: ignore

    biomlPlugin.addBlock(featureExtractionBlock)

    from Blocks.FeatureSelection import featureSelectionBlock  # type: ignore

    biomlPlugin.addBlock(featureSelectionBlock)

    from Blocks.ModelTraining import featureSelectionBlock  # type: ignore

    biomlPlugin.addBlock(featureSelectionBlock)
