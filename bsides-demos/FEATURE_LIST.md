# Network Feature List

This list includes the features extracted from network traffic used to train the DoH tunnel detection models. 
You can use this as a basis for executing the Target Zoo Attack. Ensure that your code accurately extracts these 
features and utilizes the correct indexes.

0 -> FlowBytesSent
1 -> FlowSentRate
2 -> FlowBytesReceived 
3 -> FlowReceivedRate    
4 -> PacketLengthVariance
5 -> PacketLengthStandardDeviation
6 -> PacketLengthMean
7 -> PacketLengthMedian
8 -> PacketLengthMode  
9 -> PacketLengthSkewFromMedian  
10 -> PacketLengthSkewFromMode
11 -> PacketLengthCoefficientofVariation
12 -> PacketTimeVariance
13 -> PacketTimeStandardDeviation
14 -> PacketTimeMean
15 -> PacketTimeMedian
16 -> PacketTimeMode
17 -> PacketTimeSkewFromMedian
18 -> PacketTimeSkewFromMode
19 -> PacketTimeCoefficientofVariation
20 -> ResponseTimeVariance
21 -> ResponseTimeStandardDeviation
22 -> ResponseTimeMean
23 -> ResponseTimeMedian
24 -> ResponseTimeMode
25 -> ResponseTimeSkewFromMedian
26 -> ResponseTimeSkewFromMode
27 -> ResponseTimeCoefficientofVariation

