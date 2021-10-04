import numpy as np
from scipy import signal
from sleepstages.parameterSetup import ParameterSetup

from .featureExtractor import FeatureExtractor


class FeatureExtractorWavelet(FeatureExtractor):
    extractorType = "wavelet"

    def get_features(
        self, eegSegment, timeStampSegment=0, time_step=0, local_mu=0, local_sigma=0
    ) -> np.ndarray:
        params = ParameterSetup()
        widths = params.waveletWidths
        waveletTransformed = signal.cwt(eegSegment, signal.ricker, widths)
        return waveletTransformed
