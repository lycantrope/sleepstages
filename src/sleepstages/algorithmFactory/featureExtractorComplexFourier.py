import numpy as np
from scipy import signal
from .featureExtractor import FeatureExtractor


class FeatureExtractorComplexFourier(FeatureExtractor):
    extractorType = "complexFourier"

    def get_features(
        self, eegSegment, timeStampSegment=0, time_step=0, local_mu=0, local_sigma=0
    ) -> np.ndarray:
        complexFourierMat = signal.cft(eegSegment)
        return complexFourierMat
