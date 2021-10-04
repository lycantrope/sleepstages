import numpy as np
from .featureExtractor import FeatureExtractor


class FeatureExtractorRawData(FeatureExtractor):
    extractorType = "rawData"

    def get_features(
        self, eegSegment, timeStampSegment, time_step, local_mu, local_sigma
    ) -> np.ndarray:
        return eegSegment
