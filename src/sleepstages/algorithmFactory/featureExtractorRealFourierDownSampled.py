import numpy as np
from sleepstages.parameterSetup import ParameterSetup

from .featureExtractor import FeatureExtractor

from .featureDownSampler import FeatureDownSampler


class FeatureExtractorRealFourierDownSampled(FeatureExtractor):
    extractorType = "realFourier-downsampled"

    def __init__(self):
        params = ParameterSetup()
        self.outputDim = params.downsample_outputDim

    def get_features(
        self, eegSegment, timeStampSegment=0, time_step=0, local_mu=0, local_sigma=0
    ) -> np.ndarray:
        fourierTransformed = np.fft.fft(eegSegment)
        inputTensor = np.array([[fourierTransformed]])
        fourierDownsampled = FeatureDownSampler.downSample(inputTensor, self.outputDim)[
            0
        ]
        return fourierDownsampled
