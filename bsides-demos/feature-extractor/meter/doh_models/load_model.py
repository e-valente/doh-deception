from joblib import load
import pandas as pd
import numpy as np


def load_model(model_input_dir: str):
    model_filename = 'GradientBoosting-BCCC-CIRA-DoH-Brw-2020.joblib'
    model = load(model_input_dir + model_filename)
    return model

def load_model1(model_input_dir: str):
    model_filename = 'GradientBoosting-BCCC-CIRA-DoH-Brw-2020.joblib'
    model = load(model_input_dir + model_filename)
    X = load_data("../../../sbseg-experiments/")
    model_predict(model, X)


def load_data(data_input_dir: str):
    # HKD dataset data
    X_features_dnstt_file = data_input_dir + 'X-malicious-dnstt.csv'
    X_features_dnstt = pd.read_csv(X_features_dnstt_file, sep=',').drop(columns=['Unnamed: 0'])
    return np.array(X_features_dnstt)


def model_predict(model, data):
    pred = model.predict(data[1].reshape(1, -1))

    # prints data
    print("Data: ", data[0])
    if pred[0] == 1:
        print(f'Prediction: Malicious')
    else:
        print(f'Prediction: Benign')
