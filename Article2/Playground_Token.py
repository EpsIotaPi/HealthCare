import spacy



text = "The Nile is a major north-flowing river in Northeastern Africa."
nlp = spacy.load('en_core_web_sm', exclude=['parser', 'ner'])
english_tagger_pipeline = spacy.load('en_dual_none_contextual')
nlp.add_pipe('pymusas_rule_based_tagger', source=english_tagger_pipeline)
print(nlp.pipeline)
doc = nlp(text)
# print(doc.vector.shape)

for token in doc:
    print(token.text, token._.pymusas_mwe_indexes)

for token in doc:
    start, end = token._.pymusas_mwe_indexes[0]
    if (end - start) > 1:
        print(f'{token.text}\t{token.pos_}\t{(start, end)}\t{token._.pymusas_tags}')
