

from streamlit.web import bootstrap

real_script = 'app.py'
bootstrap.run(real_script, f'run.py {real_script}', [], {})

import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import numpy as np
#
# # Load the language model
# nlp = spacy.load('en_core_web_md')
#
# # Example sentences
# sentences = ["Trump angry over election results.",
#              "the second.",
#              "This pardon the third Trump.",
#              "biden wins election and trump contests results"]
#
# # Create document objects for each sentence
# docs = [nlp(sentence) for sentence in sentences]
#
# # Initialize a matrix to store similarity scores
# similarity_matrix = np.zeros((len(docs), len(docs)))
#
# # Fill the matrix with similarity scores
# for i in range(len(docs)):
#     for j in range(len(docs)):
#         similarity_matrix[i, j] = docs[i].similarity(docs[j])
#
# # Plotting the similarity matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(similarity_matrix, annot=True, cmap='jet', xticklabels=sentences, yticklabels=sentences)
# plt.title('Sentence Similarity Matrix')
# plt.show()
