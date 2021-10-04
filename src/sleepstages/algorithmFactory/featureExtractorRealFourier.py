import numpy as np
from sleepstages.parameterSetup import ParameterSetup

from .featureExtractor import FeatureExtractor


class FeatureExtractorRealFourier(FeatureExtractor):
    extractorType = "realFourier"

    def get_features(
        self, eegSegment, timeStampSegment=0, time_step=0, local_mu=0, local_sigma=0
    ) -> np.ndarray:
        fourierTransformed = np.fft.fft(eegSegment)
        return np.array([fourierTransformed])
