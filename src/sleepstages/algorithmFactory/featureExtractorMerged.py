import numpy as np
from typing import List
from sleepstages.parameterSetup import ParameterSetup

from .featureExtractor import FeatureExtractor


class FeatureExtractorMerged(FeatureExtractor):
    def __init__(self, extractorType, extractors: List[FeatureExtractor], param=None):
        self.extractorType = extractorType
        self.extractors = extractors
        params = ParameterSetup()
        self.outputDim = params.downsample_outputDim

    def get_features(
        self, eegSegment, timeStampSegment=0, time_step=0, local_mu=0, local_sigma=0
    ) -> np.ndarray:

        merged = np.concatenate(
            [extractor.get_features(eegSegment) for extractor in self.extractors],
            axis=0,
        )
        # print('****** merged.shape = ' + str(merged.shape))
        # print('########### final merged.shape = ' + str(merged.shape))
        mergedTransposed = merged.transpose()
        return mergedTransposed
