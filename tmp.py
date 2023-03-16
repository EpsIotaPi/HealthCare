
import spacy



def get_usas_feature():
    nlp = spacy.load('en_core_web_sm', exclude=['parser', 'ner'])
    english_tagger_pipeline = spacy.load('en_dual_none_contextual')
    nlp.add_pipe('pymusas_rule_based_tagger', source=english_tagger_pipeline)
    return nlp



