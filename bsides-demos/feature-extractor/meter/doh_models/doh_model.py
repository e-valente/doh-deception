from joblib import load
import numpy as np
from sklearn.preprocessing import normalize

features = {
    'SourceIP': False,
    'DestinationIP': False,
    'SourcePort': False,
    'DestinationPort': False,
    'TimeStamp': False,
    'Duration': False,
    'FlowBytesSent': True,
    'FlowSentRate': True,
    'FlowBytesReceived': True,
    'FlowReceivedRate': True,
    'PacketLengthVariance': True,
    'PacketLengthStandardDeviation': True,
    'PacketLengthMean': True,
    'PacketLengthMedian': True,
    'PacketLengthMode': True,
    'PacketLengthSkewFromMedian': True,
    'PacketLengthSkewFromMode': True,
    'PacketLengthCoefficientofVariation': True,
    'PacketTimeVariance': True,
    'PacketTimeStandardDeviation': True,
    'PacketTimeMean': True,
    'PacketTimeMedian': True,
    'PacketTimeMode': True,
    'PacketTimeSkewFromMedian': True,
    'PacketTimeSkewFromMode': True,
    'PacketTimeCoefficientofVariation': True,
    'ResponseTimeTimeVariance': True,
    'ResponseTimeTimeStandardDeviation': True,
    'ResponseTimeTimeMean': True,
    'ResponseTimeTimeMedian': True,
    'ResponseTimeTimeMode': True,
    'ResponseTimeTimeSkewFromMedian': True,
    'ResponseTimeTimeSkewFromMode': True,
    'ResponseTimeTimeCoefficientofVariation': True
}


def load_model(model_input_dir: str):
    # model_filename = 'GradientBoosting-BCCC-CIRA-DoH-Brw-2020.joblib'
    model_filename = 'GradientBoosting-e-valente-customized.joblib'
    print(f"loading the model {model_filename}")
    model = load(model_input_dir + model_filename)
    return model


def normalize_data(data):
    return normalize(data, norm='l2', axis=1)


def extract_features(data):
    # target_features is a numpy array
    target_features = np.array([], dtype=np.float64)

    for key in features:
        if features[key]:
            # print(f"{key}: {data[key]}")
            # concatenate all feature separated by comma
            target_features = np.concatenate((target_features, data[key]), axis=None)

    #print(target_features)
    return target_features


class DoH:
    def __init__(self, model_input_dir: str):
        self.model = load_model(model_input_dir)

    def predict(self, data):
        print("extracting features...")
        # target_features is a numpy array

        target_features = extract_features(data)

        # check if it has a nan values in the array
        if np.isnan(target_features).any():
            print('Data contains NaN values. Please check the data.')
            return

        #print('before normalization:', target_features)
        target_features = normalize_data(target_features.reshape(1, -1))
        #print('after normalization:', target_features)

        pred = self.model.predict(target_features.reshape(1, -1))

        if pred[0] == 1:
            print(f' ============> Prediction: Malicious')
        else:
            print(f'=============> Prediction: Benign')
