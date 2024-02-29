import time
import torch
from matplotlib import pyplot as plt
from pykeops.torch import LazyTensor
import numpy as np

from ray.kmean_torch import kmeans_core



train, test, validation = np.load("hubert_features/librispeech_asr/train.clean.100.npy"), np.load("hubert_features/librispeech_asr/test.clean.npy"), np.load("hubert_features/librispeech_asr/validation.clean.npy")


km = kmeans_core(100,train,batch_size=262144)
outputs = km.run()
