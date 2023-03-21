
import spacy
import numpy as np

from .feature_tokens import usas_feature_list


class FeatureUSAS():
    def __init__(self):
        self.usas_feature_dict = {}
        for idx, t in enumerate(usas_feature_list):
            self.usas_feature_dict[t] = idx

        self.nlp = spacy.load('en_core_web_sm', exclude=['parser', 'ner'])
        english_tagger_pipeline = spacy.load('en_dual_none_contextual')
        self.nlp.add_pipe('pymusas_rule_based_tagger', source=english_tagger_pipeline)

        self.n_feature = len(usas_feature_list)

    def get_feature(self, text, weight_level="token", normalize=True):
        """

        :param text:
        :param weight_level: assign weights based on token or word
        :param normalize: normalize based on text length or not
        :return: features from text
        """
        feature = np.zeros(self.n_feature)
        doc = self.nlp(text)
        for i, word in enumerate(doc):
            tokens = []
            for tag in word._.pymusas_tags:
                tokens.extend(self.filter(tag))
            for t in tokens:
                if weight_level == "token":
                    feature[t] += 1                # Token have independent weight
                elif weight_level == "word":
                    feature[t] += 1 / len(tokens)  # Token sharing weights for the same word

        if normalize:
            feature = feature / len(text)
        return feature

    def filter(self, token):
        """
        :return: Token after merging into 115 classes
        example:
                1. N3.2 -> [N3]
                2. Z1mf -> [Z1], N5+ -> [N5]
                3. W3/M4 -> [W3, M4]
                4. PUNCT -> []
        """
        other_symbol = ['%', '@', 'f', 'm', 'c', 'n', 'i', '-', '+']
        token = token.split('/')
        result_token = []
        for i in range(len(token)):
            t = token[i]
            t = "".join(list(filter(lambda x: x not in other_symbol, t)))  # remove characters from the other_symbol
            tok = t.split('.')[0]
            if tok in self.usas_feature_dict.keys():
                result_token.append(self.usas_feature_dict[tok])
        return result_token

