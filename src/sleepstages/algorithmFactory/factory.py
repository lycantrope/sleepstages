import importlib
from typing import Dict
from .featureExtractor import FeatureExtractor

EXTRACTORS: Dict[str, str] = {
    "classical": "featureExtractorClassical",
    "freqHisto": "featureExtractorFreqHisto",
    "freqHistoWithTime": "featureExtractorFreqHistoWithTime",
    "wavelet": "featureExtractorWavelet",
    "wavelet-downsampled": "featureExtractorWaveletDownSampled",
    "realFourier": "featureExtractorRealFourier",
    "realFourier-downsampled": "featureExtractorRealFourierDownSampled",
    "complexFourier": "featureExtractorComplexFourier",
    "rawData": "featureExtractorRawData",
    "rawDataWithFreqHistoWithTime": "featureExtractorRawDataWithFreqHistoWithTime",
    "rawDataWithSTFTWithTime": "featureExtractorRawDataWithSTFTWithTime",
}


class AlgorithmFactory:
    @staticmethod
    def generate_extractor(extractor_types) -> FeatureExtractor:
        if "," not in extractor_types and extractor_types not in EXTRACTORS:
            raise NotImplementedError(f"Extractor '{extractor_types}' not available.")

        if extractor_types in EXTRACTORS:
            module_name = EXTRACTORS[extractor_types]
            module = importlib.import_module(
                "." + module_name, package="sleepstages.algorithmFactory"
            )
            extractor = getattr(module, module_name.replace("feature", "Feature"))()
            return extractor

        if "," in extractor_types:
            print("Using FeatureExtractorMerged...")
            extractors = [
                AlgorithmFactory.generate_extractor(extractor_type.strip())
                for extractor_type in extractor_types.split(",")
            ]

            module = importlib.import_module(
                ".featureExtractorMerged", package="sleepstages.algorithmFactory"
            )
            extractor = module.FeatureExtractorMerged(
                extractorType=extractor_types, extractors=extractors
            )
            return extractor
