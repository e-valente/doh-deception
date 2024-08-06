import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

# features dictionary has all key value set to True
# this is a dictionary of features that are extracted from the network data

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


def extract_features(data):
    print("extracting features...")
    # target_features is a numpy array
    target_features = np.array([], dtype=np.float64)

    for key in features:
        if features[key]:
            print(f"{key}: {data[key]}")
            # concatenate all feature separated by comma
            target_features = np.concatenate((target_features, data[key]), axis=None)

    print(target_features)
