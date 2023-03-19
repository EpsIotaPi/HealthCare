
import spacy



def get_usas_feature():
    nlp = spacy.load('en_core_web_sm', exclude=['parser', 'ner'])
    english_tagger_pipeline = spacy.load('en_dual_none_contextual')
    nlp.add_pipe('pymusas_rule_based_tagger', source=english_tagger_pipeline)
    return nlp


token_list = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15',
              'B1', 'B2', 'B3', 'B4', 'B5', 'C1', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'F1', 'F2', 'F3', 'F4',
              'G1', 'G2', 'G3', 'H1', 'H2', 'H3', 'H4', 'H5', 'I1', 'I2', 'I3', 'I4',
              'K1', 'K2', 'K3', 'K4', 'K5', 'K6', 'L1', 'L2', 'L3', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8',
              'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'O1', 'O2', 'O3', 'O4', 'P1', 'Q1', 'Q2', 'Q3', 'Q4',
              'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'T1', 'T2', 'T3', 'T4',
              'W1', 'W2', 'W3', 'W4', 'W5', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9',  'Y1', 'Y2',
              'Z0', 'Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7', 'Z8', 'Z9', 'Z99', 'PUNCT', 'altogether', 'D', 'UNKNOW']
token_dict = {}
for idx, t in enumerate(token_list):
    token_dict[t] = idx

def feature_filter(token):
    other_symbol = ['%', '@', 'f', 'm', 'c', 'n', 'i', '-', '+']
    tokens = token.split('/')
    for i in range(len(tokens)):
        t = tokens[i]
        t = "".join(list(filter(lambda x: x not in other_symbol, t)))  # 去除other_symbol中存在的元素
        tok = t.split('.')[0]
        if tok not in token_dict.keys():
            print(tok)
            tok = "UNKNOW"
        tokens[i] = token_dict[tok]
    return tokens
