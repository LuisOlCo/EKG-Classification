# EKG-Classification

Objective of the Project

The main goal of this project is the classification of EKG or ECG waves (the heartbeat waves measure by a monitor), each EKG corresponds to a patient that may have or may not have a heart disease. Both training data and test data can be classified into four different classes:

•	N: Normal beat
•	S: Supraventricular premature beat
•	V: Premature ventricular contraction beat

Data

Each sample corresponds to a full EKG wave, represented by a vector, each vector has 188 points, the first 187 represent the waveshape of the heartbeat, while the last element in the vector represents the label given to that waveshape.

•	N = 0
•	S = 1
•	V = 2

Tools and Approach

In order to perform this classification, it has been decided to explore the use different tools to see the performing of each of them, the tools that have been used for this purpose are:

•	Kn - Nearest – Neighbor Estimation
•	Feature extraction and Neural Network
•	LSTM algorithm 
