
# from yellowbrick.text import DispersionPlot, dispersion
# from yellowbrick.datasets import load_hobbies
#
# # Load the text data
# corpus = load_hobbies()
# print(corpus.data)
#
# # # Create a list of words from the corpus text
# # text = [doc.split() for doc in corpus.data]
# #
# # # Choose words whose occurence in the text will be plotted
# # target_words = ['features', 'mobile', 'cooperative', 'competitive', 'combat', 'online']
# #
# # # Create the visualizer and draw the plot
# # dispersion(target_words, text, colors=['olive'])


import spacy
import pandas as pd

nlp = spacy.load('en_core_web_sm', exclude=['parser', 'ner'])
english_tagger_pipeline = spacy.load('en_dual_none_contextual')
nlp.add_pipe('pymusas_rule_based_tagger', source=english_tagger_pipeline)


path = "/Users/jinchenji/Developer/JetBrains/Pycharm/healthcare/datasets/Article2/main.csv"

csv_data = pd.read_csv(path)

# print(csv_data["Label (1=Mistake present, 0= No mistake)"][2])

text = csv_data["Text"][2]
# print(text)

output_doc = nlp(text)

print(f'Text\t\tLemma\t\tPOS\t\tUSAS Tags')
for token in output_doc:
    print(f'{token.text}\t\t{token.lemma_}\t\t{token.pos_}\t\t{token._.pymusas_tags}')