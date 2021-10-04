import pickle
import numpy as np


class FeatureDownSampler:
    @staticmethod
    def featureDownSampling(
        inputFeaturePath: str, outputFeaturePath: str, outputDim
    ) -> None:
        "Downsample the feature from pickle file"
        print("inputFeaturePath = " + inputFeaturePath)
        print("outputFeaturePath = " + outputFeaturePath)
        with open(inputFeaturePath, "rb") as inputFileHandler:
            inputTensor = pickle.load(inputFileHandler)

        outputTensor = FeatureDownSampler.downSample(inputTensor, outputDim)
        with open(outputFeaturePath, "wb") as outputFileHandler:
            pickle.dump(outputTensor, outputFileHandler)

    @staticmethod
    def downSample(inputTensor: np.ndarray, outputDim: int) -> np.ndarray:
        # print('inputTensor.shape = ' + str(inputTensor.shape))
        inputDim = inputTensor.shape[-1]
        if inputDim == outputDim:
            # print('not downsampling')
            outputTensor = inputTensor
        else:
            poolingSize = np.int64(np.floor(np.float64(inputDim / outputDim)))
            poolingStrideSize = poolingSize
            # print('poolingSize = ' + str(poolingSize))
            # downsample by appling max(arg()) to regions
            outputTensor = np.zeros(
                (inputTensor.shape[0], inputTensor.shape[1], outputDim)
            )
            for outputIDstart in range(outputDim):
                inputIDstart = outputIDstart * poolingStrideSize
                inputIDs = range(inputIDstart, inputIDstart + poolingSize - 1)
                # print('outputIDstart = ' + str(outputIDstart) + ', inputIDs = ' + str(inputIDs))
                outputTensor[:, :, outputIDstart] = np.max(
                    inputTensor[:, :, inputIDs], axis=-1
                )
                # outputTensor[:,:,outputIDstart] = np.mean(np.abs(inputTensor[:,:,inputIDs]), axis=-1)
        # print('outputTensor.shape = ' + str(outputTensor.shape))
        # print('outputTensor = ' + str(outputTensor))
        return outputTensor
