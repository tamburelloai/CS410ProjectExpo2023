# README

# ****Visualized Media Bias and Polarization Detection****

### TEAM

Michael Tamburello (Captain)

![Untitled](./bias_image.jpeg)

## **Overview**

A visually intuitive media analysis tool designed 
to detect and visualize media bias and polarization.

This project leverages advanced Natural Language Processing (NLP) 
technologies, such as SpaCy, to parse news content from various 
outlets with known political bias. 

The app quantifies ideological biases and narrative divergences
in reporting, presenting users with dynamic visualizations that
highlight language differences and portrayals of current events.


---



## Watch the Demo 📺
Here’s a demonstration of the application in action!
(The link below directs you to a public google drive link)  

[CLICK HERE TO WATCH THE DEMO VIDEO](https://drive.google.com/file/d/1KT-lymmRFjhcGkJhKsq-jSIljQvS_jmL/view)
---

## Documentation 📘

Documentation for all classes and methods can be found in the sidebar of the application 

---

## **Setup (Cloud Hosted) ☁️ [RECOMMENDED]**

The app is already running! (hosted by streamlit) 🎉

Simply click the link below to experience a live demonstration:

[https://410project.streamlit.app/](https://410project.streamlit.app/)

---

## **Setup (Local) 📜**

0. Download or clone this repository


1. Open terminal and navigate to the parent directory of the application.

```
cd where/you/downloaded/or/cloned/to/CS410ProjectExpo2023-main
```

1. Ensure python environment you'll be using has all required dependencies installed:

```
pip install -r requirements.txt
```

1. Initialize the application easily by executing the following command:

```
streamlit run app.py
```

---

## Under the Hood

### Word2Vec

Word2Vec is a technique in natural language processing for learning word embeddings from a text corpus.
Its neural network architecture typically uses either 
Continuous Bag of Words (CBOW) or Skip-gram models. 
The core equation for Word2Vec is:


$$
P(w_{\text{context}} | w_{\text{target}}) = \frac{\exp(v_{\text{context}} \cdot v_{\text{target}})}{\sum_{w \in W} \exp(v_w \cdot v_{\text{target}})}
$$

where *v* represents word vectors, and *w* are the words in the vocabulary.This equation models the probability of a context word given a target word, learning representations that capture semantic meanings and relationships between words.

### Sentiment Analysis

In sentiment analysis, a neural network is trained to 
classify text (like sentences or titles) as positive, negative, 
or neutral. For inference, it uses a function like:

$$
y = \sigma(Wx + b)
$$

 where *y*
is the sentiment classification and  *x* represents input features (vectorized word representations).
### Transformer Architecture

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

In the context of spaCy, transformers are used to process text by encoding it into high-dimensional space, capturing complex syntactic and semantic relationships. The model, pre-trained on large datasets, efficiently learns contextualized word embeddings. These embeddings are then fine-tuned for specific tasks in NLP, such as named entity recognition, part-of-speech tagging, and sentiment analysis.

### t-SNE

t-SNE (t-Distributed Stochastic Neighbor Embedding) is a technique for dimensionality reduction, particularly useful for visualizing high-dimensional data. The core equation involves calculating the conditional probabilities:

$$
p_{j|i} = \frac{\exp\left(-\frac{\|x_i - x_j\|^2}{2\sigma_i^2}\right)}{\sum_{k \neq i} \exp\left(-\frac{\|x_i - x_k\|^2}{2\sigma_i^2}\right)}
$$

t-SNE aims to find a low-dimensional representation that preserves the neighborhood structure of the data.

## Self-Evaluation

The majority of what I set out to achieve has been accomplished, especially in the development of the application and the implementation of its key functionalities. Inspired by Google's embedding projector, the project's goal was to offer a nuanced view of word relationships through reduced dimensional representations. Specifically their distances in the embedded space. This objective has been met.

I acknowledge that the effectiveness of the projector could be improved. While the current implementation provides insightful visualizations, its depth and clarity could be enhanced with more extensive filtering mechanisms.

In conclusion, while the central idea of the app has been successfully implemented and its use case validated, I recognize that there is room for refinement. The application stands as a functional and educational tool, yet I am aware of its potential for further development.
