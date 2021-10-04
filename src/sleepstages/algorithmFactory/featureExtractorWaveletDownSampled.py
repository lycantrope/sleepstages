import numpy as np
from sleepstages.parameterSetup import ParameterSetup

from scipy import signal

from .featureDownSampler import FeatureDownSampler
from .featureExtractor import FeatureExtractor


class FeatureExtractorWaveletDownSampled(FeatureExtractor):
    extractorType = "wavelet-downsampled"

    def __init__(self):
        params = ParameterSetup()
        self.outputDim = params.downsample_outputDim

    def get_features(
        self, eegSegment, timeStampSegment=0, time_step=0, local_mu=0, local_sigma=0
    ) -> np.ndarray:
        params = ParameterSetup()
        widths = params.waveletWidths
        waveletTransformed = signal.cwt(eegSegment, signal.ricker, widths)
        inputTensor = np.array([waveletTransformed])
        # print('inputTensor.shape = ' + str(inputTensor.shape))
        # print('self.outputDim = ' + str(self.outputDim))
        waveletTransformedDownsampled = FeatureDownSampler.downSample(
            inputTensor, self.outputDim
        )[0]
        # print('waveletTransformedDownsampled.shape = ' + str(waveletTransformedDownsampled.shape))
        return waveletTransformedDownsampled
