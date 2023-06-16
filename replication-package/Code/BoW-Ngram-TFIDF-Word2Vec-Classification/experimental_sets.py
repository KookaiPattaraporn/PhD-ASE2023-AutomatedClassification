import pandas as pd
import numpy as np
import load_data


def create_file(project):
    training_size = 60
    validation_size = 20
    test_size = 20

    dataframe = load_data.dataset(project, '')
    data = dataframe.values
    print('No. of issue: ' + str(len(data)))

    if training_size + validation_size + test_size == 100:

        numData = len(data)
        numTrain = int((training_size * numData) / 100)
        numValidation = int((validation_size * numData) / 100)
        numTest = int((test_size * numData) / 100)

        print("#Total size: %s" % numData)
        print("#Training : %s, #Validation : %s, #Testing : %s" % (numTrain, numValidation, numTest))
        print("Total: %s" % (numTrain + numValidation + numTest))

        firstStop = numTrain
        secondStop = numTrain + numValidation

        divided_set = np.zeros([numData, 3]).astype(int)
        divided_set[0:firstStop, 0] = 1
        divided_set[firstStop:secondStop, 1] = 1
        divided_set[secondStop:numData, 2] = 1

        setsDataFrame = pd.DataFrame(divided_set, columns = ['train', 'validate', 'test'])
        setsDataFrame.to_csv('Data/' + project + '_experimental_sets.csv', index = False)

        print('Writing experimental sets successfully')

        validVector = dataframe.iloc[firstStop:secondStop, 4:]
        validVector.to_csv('Data/' + project + '_valid_actual.csv', index = False, header = False)
        testVector = dataframe.iloc[secondStop:numData, 4:]
        testVector.to_csv('Data/' + project + '_actual.csv', index = False, header = False)
        print('Writing actual successfully')

    else:
        print('Invalid experimental setting')
