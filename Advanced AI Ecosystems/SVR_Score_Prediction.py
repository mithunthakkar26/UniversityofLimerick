# Below code is referenced from https://github.com/ScalarPy/AWS-Sagemaker-Deploy

from __future__ import print_function

import argparse
import os
import pandas as pd

from sklearn import tree
from sklearn.externals import joblib

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Sagemaker specific arguments. Defaults are set in the environment variables.

    #Saves Checkpoints and graphs
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])

    #Save model artifacts
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])

    #Train data
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    file = os.path.join(args.train, "Anime_Preprocessed.csv")
    dataset = pd.read_csv(file, engine="python")

    # labels are in the first column
    X = dataset.iloc[:, 1:].values
    y = dataset.iloc[:, 0].values

        
    from sklearn.svm import SVR
    regressor = SVR(C = 10, gamma = 0.001)
    regressor.fit(X, y)

    # Print the coefficients of the trained classifier, and save the coefficients
    joblib.dump(regressor, os.path.join(args.model_dir, "model.joblib"))
    
    
def model_fn(model_dir):
    """Deserialized and return fitted model
    
    Note that this should have the same name as the serialized model in the main method
    """
    regressor = joblib.load(os.path.join(model_dir, "model.joblib"))
    return regressor