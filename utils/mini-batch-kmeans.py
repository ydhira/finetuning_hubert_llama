import numpy as np
from sklearn.cluster import MiniBatchKMeans

train, test, validation = np.load("hubert_features/librispeech_asr/train.clean.100.npy"), np.load("hubert_features/librispeech_asr/test.clean.npy"), np.load("hubert_features/librispeech_asr/validation.clean.npy")

kmeans =  MiniBatchKMeans(
    n_clusters=100, 
    init="k-means++",
    batch_size=524288,
    verbose=1,
    random_state=42,
    max_iter=100,
    max_no_improvement=5,
)

outputs = kmeans.fit(train)


import pdb;pdb.set_trace()