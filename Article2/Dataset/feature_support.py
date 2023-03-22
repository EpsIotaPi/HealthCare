import numpy as np



usas_support_5 = [False, True, False, False, False, False, False, False, False, False, True, False, False, False, False,
                  False, True, False, False, False, False, False, False, False, True, False, False, False, False, False,
                  False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                  False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                  False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                  False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                  False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                  False, False, False, False, False, False, False, False, False, True, False, False, False, False, False]



supports_dict = {
    'usas': usas_support_5,
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