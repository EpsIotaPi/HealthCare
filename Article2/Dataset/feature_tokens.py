


# complex_sentences_6 = ["Average number of sentences per paragraph",
#                         "Number of difficult sentences (more than 22 words)",
#                         "Longest sentence (sentence #2)", "Average sentence length",
#                         "Passive voice", "Sentences that begin with conjunctions"]
# lexical_complexity_3 = ["Number of unique words", "Number of unique long words",
#                         "Number of unique monosyllabic words"]
# morph_ortho_complexity_8 = ["Number of syllables", "Average number of characters",
#                             "Average number of syllables", "Number of monosyllabic words",
#                             "Number of complex (3+ syllable) words", "Number of unique 3+ syllable words",
#                             "Number of long (6+ characters) words", "Misspellings"]
# content_density_3 = ["Number of proper nouns",
#                      "Overused words (x sentence)",
#                      "Wordy items"]

# structural_feature_list = complex_sentences_6 + lexical_complexity_3 + morph_ortho_complexity_8 + content_density_3

usua_feature_list = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15',
                     'B1', 'B2', 'B3', 'B4', 'B5', 'C1', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'F1', 'F2', 'F3', 'F4',
                     'G1', 'G2', 'G3', 'H1', 'H2', 'H3', 'H4', 'H5', 'I1', 'I2', 'I3', 'I4',
                     'K1', 'K2', 'K3', 'K4', 'K5', 'K6', 'L1', 'L2', 'L3', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8',
                     'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'O1', 'O2', 'O3', 'O4', 'P1', 'Q1', 'Q2', 'Q3', 'Q4',
                     'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'T1', 'T2', 'T3', 'T4',
                     'W1', 'W2', 'W3', 'W4', 'W5', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9',  'Y1', 'Y2',
                     'Z0', 'Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7', 'Z8', 'Z9', 'Z99']

# all_features_list = structural_feature_list + usua_feature_list


feature_dict = {
    "usua": usua_feature_list,
}