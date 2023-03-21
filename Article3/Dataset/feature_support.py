import numpy as np

TOF_support_5 = [True, True, False, False, False, False, True, False, False, True,
                 False, False, False, False, False, False, False, False, False, True]

SOF_support_14 = [False, True, True, True, False, True, True, False, False, False, False, False, True, False, False,
                  True, True, True, False, False, False, False, False, False, False, False, False, False, False, False,
                  False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                  False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                  False, False, False, False, False, False, True, False, True, True, False, False, False, False,
                  False, False, False, False, False, False, False, False, False, False, False, False, False,
                  False, False, False, False, False, False, False, False, False, False, False, False, False,
                  False, False, False, False, False, False, False, False, False, False, False, True, False,
                  False, False, False, True]

JOF_support_57 = [False, True, True, True, True, False, True, True, True, False, False, False, True, False, True, True,
                  True, True, True, True, True, True, True, True, True, True, True, False, True, True, False, False,
                  True, False, False, True, True, True, True, True, True, False, False, False, False, False, False,
                  True, False, False, False, False, False, False, False, False, False, False, False, False, False,
                  False, False, False, False, False, False, False, False, True, True, True, False, True, False, False,
                  False, True, True, False, True, False, True, False, True, True, True, True, False, False, True, True,
                  True, False, False, True, True, False, False, False, False, True, True, False, False, False, False,
                  False, False, False, False, False, False, False, True, True, False, False, False, False, False, True,
                  False, False, False, False, False, False, False, True, True, False, True, False, True]


supports_dict = {
    'JOF': JOF_support_57,
    'TOF': TOF_support_5,
    'SOF': SOF_support_14,
}

def select_features(features, support):
    new_feat = np.zeros_like(features)
    true_count = 0
    for i in range(len(support)):
        if support[i]:
            new_feat[:, true_count] = features[:, i]
            true_count += 1
    new_feat = new_feat[:, :true_count]
    return new_feat

# test = np.random.rand(10, 135)
# feat = select_feature(features=test, support=JOF_support_57)
#
# print(feat.shape)