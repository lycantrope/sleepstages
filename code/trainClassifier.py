from __future__ import print_function
import sys
from os import listdir
from parameterSetup import ParameterSetup
from classifierTrainer import trainClassifier

args = sys.argv
optionType = args[1] if len(args) > 2 else ''
optionVals = args[2:] if len(args) > 2 else [0]
print('optionType =', optionType, ', optionVals = ', optionVals)
paramDir = '../data/compare'
outputDir = ''
for paramFileName in listdir(paramDir):
    if paramFileName.startswith('params.') and not paramFileName.endswith('~'):
        print('paramFileName =', paramFileName)
        params = ParameterSetup(paramDir, paramFileName, outputDir)
        trainClassifier(params, outputDir, optionType, optionVals)
