import spacy
import asent

# load spacy pipeline
nlp = spacy.blank('en')
nlp.add_pipe('sentencizer')

# add the rule-based sentiment model
nlp.add_pipe('asent_en_v1')

# try an example
text = 'I am not very happy, but I am also not especially sad'
doc = nlp(text)

# print polarity of document, scaled to be between -1, and 1
print(doc._.polarity)
# neg=0.0 neu=0.631 pos=0.369 compound=0.7526

# Naturally, a simple score can be quite unsatisfying, thus Asent implements a series of visualizer to interpret the results:
print(type(asent.visualize(doc, style='prediction')))
 # or
