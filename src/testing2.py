import spacy
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd


# Load spaCy model
nlp = spacy.load('en_core_web_md')  # or another model

# Example text data
texts = ['cat', 'dog', 'apple', 'orange', 'car', 'bike']
og_texts = list(texts)
texts.extend("Utilizes cutting-edge Natural Language Processing (NLP) techniques to analyze and visualize media bias across various news outlets. The platform's primary function is to identify and quantify ideological biases and narrative divergences in news reporting, using advanced tools like spaCy and PyTorch. Users will be able to see dynamic visualizations that highlight differences in media language and portrayal of current events, offering a clearer understanding of media polarization. The platform aims to enhance public dialogue and support democratic values by making users aware of the biases in news sources. The app's effectiveness will be evaluated through precision, recall, F1-score, ROC-AUC metrics, and user feedback.".split(' '))

# Get embeddings
embeddings = np.array([nlp(text).vector for text in texts])

# Dimensionality reduction with t-SNE
tsne = TSNE(n_components=3, random_state=0)
data = tsne.fit_transform(embeddings)
import plotly.express as px

data_min_max_scaled = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))

x = data_min_max_scaled[:, 0].reshape(-1)
y = data_min_max_scaled[:, 1].reshape(-1)
z = data_min_max_scaled[:, 2].reshape(-1)
labels = [np.random.choice(['A', 'B']) for i in range(len(x))]

df =  pd.DataFrame({'word': texts,
                 'x': x,
                 'y': y,
                 'z': z,
                 'partisan': labels,
                 })

unique_labels = df['partisan'].unique()
colors = px.colors.qualitative.Plotly  # or any other color sequence
color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}

import plotly.graph_objects as go

# Create an empty figure
fig = go.Figure()

# Add X, Y, and Z axis as scatter plots
fig.add_trace(go.Scatter3d(x=[0, 1], y=[0, 0], z=[0, 0],
                           mode='lines+text',
                           name='X-axis',
                           text=['', 'X'],
                           textposition='top center',
                           line=dict(color='red', width=2)))

fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 1], z=[0, 0],
                           mode='lines+text',
                           name='Y-axis',
                           text=['', 'Y'],
                           textposition='top center',
                           line=dict(color='green', width=2)))

fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, 1],
                           mode='lines+text',
                           name='Z-axis',
                           text=['', 'Z'],
                           textposition='top center',
                           line=dict(color='blue', width=2)))

# Add a scatter plot for this label
fig.add_trace(go.Scatter3d(
    x=df['x'],
    y=df['y'],
    z=df['z'],
    mode='markers',
    marker=dict(
        size=4,
        color=df['partisan'].map(color_map),  # apply the color map to the 'label' column
    ),
    text=df['word']  # this will set the label text to each data point
))

# Setting the layout of the figure
fig.update_layout(
    scene=dict(
        xaxis=dict(showbackground=False, showline=False, zeroline=False, showticklabels=False, title='', showspikes=False),
        yaxis=dict(showbackground=False, showline=False, zeroline=False, showticklabels=False, title='', showspikes=False),
        zaxis=dict(showbackground=False, showline=False, zeroline=False, showticklabels=False, title='', showspikes=False),
        camera=dict(
            eye=dict(x=1.25, y=1.25, z=1.25)
        )
    ),
    margin=dict(l=0, r=0, b=0, t=0)
)

# Set the layout for a dark background
fig.update_layout(
    scene=dict(
        xaxis=dict(showbackground=False, backgroundcolor="rgb(10, 10, 10)",
                   gridcolor="rgb(10, 10, 10)", showticklabels=False),
        yaxis=dict(showbackground=False, backgroundcolor="rgb(10, 10, 10)",
                   gridcolor="rgb(10, 10, 10)", showticklabels=False),
        zaxis=dict(showbackground=False, backgroundcolor="rgb(10, 10, 10)",
                   gridcolor="rgb(10, 10, 10)", showticklabels=False),
        camera=dict(
            eye=dict(x=1.25, y=1.25, z=1.25)
        )
    ),
    paper_bgcolor='rgb(10, 10, 10)',
    plot_bgcolor='rgb(10, 10, 10)',
    margin=dict(l=0, r=0, b=0, t=0)
)

# Optionally, remove the legend if it's not needed
fig.update_layout(showlegend=False)

# Show figure
fig.show()


# Show figure
fig.show()