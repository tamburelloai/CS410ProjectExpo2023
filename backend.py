import json
import asent
import spacy
from sklearn.manifold import TSNE
from spacy import displacy
from managers.news_api_manager import NewsAPIManager
import pandas as pd
import numpy as np
from transformers import pipeline
from sklearn.decomposition import PCA
from utils import left_wing_sources, right_wing_sources

def load_secrets():
    res = {}
    with open('files/secrets.txt', 'r') as f:
        for line in f.readlines():
            line = line.split('=')
            assert len(line) == 2
            source, apikey = line
            apikey = apikey.replace('\n', '')
            res[source] = apikey
    return res

secrets = load_secrets()

class Backend:
    def __init__(self):
        """
        Initializes the Backend class.
            - Loads secrets for API access.
            - Initializes news API manager with API key.
            - Loads Spacy NLP model and sentiment model.
            - Sets up left-wing and right-wing news sources.
        """
        secrets = load_secrets()
        self.newsapi = NewsAPIManager(secrets['newsapi'])
        self.nlp_model = spacy.load("en_core_web_md")
        self.sentiment_model = spacy.blank('en')
        self.sentiment_model.add_pipe('sentencizer')
        self.sentiment_model.add_pipe('asent_en_v1')
        #
        self.leftwing_sources = left_wing_sources
        self.rightwing_sources = right_wing_sources
        self.query = None
        self.articles = None
        self.total_results = 0
        self.num_leftwing_results = 0
        self.num_rightwing_results = 0
        self.parsed_text = None


    def run_query(self, query):
        """
        Executes a news query and processes the results.
        -  If in testing mode, loads sample data.
        - Otherwise, fetches news articles from left-wing and right-wing sources.
        - Processes the query results for further analysis.
        :param query: A string representing the news query.
       """
        testing = True #False

        if testing:
            print('testing true')
            with open('samples/newsapi_leftwing_sample.json', 'r') as f:
                self.leftwing_response = json.load(f)
                self.num_leftwing_results = self.leftwing_response['totalResults']
                self.leftwing_response = pd.DataFrame(self.leftwing_response)['articles'].tolist()
            with open('samples/newsapi_rightwing_sample.json', 'r') as f:
                self.rightwing_response = json.load(f)
                self.num_rightwing_results = self.rightwing_response['totalResults']
                self.rightwing_response = pd.DataFrame(self.rightwing_response)['articles'].tolist()
        else:
            self.leftwing_response = self.newsapi(query=query, sources=', '.join(self.leftwing_sources))
            self.num_leftwing_results = self.leftwing_response['totalResults']
            self.leftwing_response = pd.DataFrame(self.leftwing_response)['articles'].tolist()

            self.rightwing_response = self.newsapi(query=query, sources=', '.join(self.rightwing_sources))
            self.num_rightwing_results = self.rightwing_response['totalResults']
            self.rightwing_response = pd.DataFrame(self.rightwing_response)['articles'].tolist()
        self.process_query_results()

    def sort_titles_by_similarity(self, leftwing_titles, rightwing_titles):
        """
        Sorts news article titles by their similarity.
        - Compares titles from left-wing and right-wing sources.
        - Pairs titles based on similarity scores.

        :param leftwing_titles: A list of strings containing titles from left-wing sources.
        :param rightwing_titles: A list of strings containing titles from right-wing sources.
        :return: Two lists of sorted titles from left-wing and right-wing sources.
        """
        # Create document objects for each sentence
        left_docs = [self.nlp_model(sentence) for sentence in leftwing_titles]
        right_docs = [self.nlp_model(sentence) for sentence in rightwing_titles]

        # Initialize a list to store the pairs
        paired_titles = []

        # Iterate over each document in left_docs with its index
        for left_index, left_doc in enumerate(left_docs):
            # Initialize a variable to store the highest similarity score and corresponding index
            max_similarity = 0
            most_similar_index = -1

            # Iterate over each document in right_docs with its index
            for right_index, right_doc in enumerate(right_docs):
                # Calculate similarity
                similarity = left_doc.similarity(right_doc)

                # Check if this is the highest similarity so far
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_index = right_index

            # Add the most similar pair to the list
            if most_similar_index != -1:
                paired_titles.append((leftwing_titles[left_index], rightwing_titles[most_similar_index]))

        left_output = [x[0] for x in paired_titles]
        right_output = [x[1] for x in paired_titles]
        return left_output, right_output

    def process_query_results(self):
        """
        Processes the results of a news query.
        - Extracts titles from left-wing and right-wing responses.
        - Sorts titles by similarity and analyzes sentiment.
        """
        assert self.leftwing_response and self.rightwing_response
        # extract titles from response
        leftwing_titles = pd.DataFrame(self.leftwing_response)['title'].tolist()
        rightwing_titles = pd.DataFrame(self.rightwing_response)['title'].tolist()
        #sort by similarity to pair documents regardless of polarity/sentiment
        self.leftwing_titles, self.rightwing_titles = self.sort_titles_by_similarity(leftwing_titles, rightwing_titles)
        # get sentiment of titles for coloring
        self.leftwing_dataframe = self.build_response_dataset(self.leftwing_titles)
        self.rightwing_dataframe = self.build_response_dataset(self.rightwing_titles)

    def map_sentiment_to_color(self, value):
        """
        Maps sentiment values to corresponding colors.
        - Currently a placeholder for future implementation.

        :param value: A sentiment value to map.
        :return: A color representation of the sentiment.
        """
        return value

    def build_response_dataset(self, titles):
        """
        Builds a dataset from news titles with associated sentiment scores.
        - Computes sentiment for each title.
        - Filters out titles with neutral sentiment.

        :param titles: A list of news titles.
        :return: A pandas DataFrame with titles and their sentiment scores.
        """
        sentiment = []
        for t in titles:
            sentiment.append(self.map_sentiment_to_color(self.get_sentiment(t)['compound']))

        final_titles, final_sentiments = [], []
        for i in range(len(sentiment)):
            if sentiment[i] != 0:
                final_titles.append(titles[i])
                final_sentiments.append(sentiment[i])
        return pd.DataFrame({'title': final_titles, 'sentiment': final_sentiments})


    def get_entity_breakdown_html(self, query_string):
        """
        Generates HTML visualization for named entities in a text.
        - Uses Spacy's displacy for rendering entities.

        :param query_string: A string to analyze for named entities.
        :return: A string containing HTML for the entity visualization.
        """
        doc = self.nlp_model(query_string)
        colors = {"ORG": "linear-gradient(90deg, #aa9cfc, #fc9ce7)"}
        options = {"colors": colors}
        ent_html = displacy.render(doc, style="ent", options=options, jupyter=False)
        return ent_html

    def get_sentiment(self, data_i):
        """
        Analyzes sentiment of a given text.
        - Uses the sentiment model to calculate sentiment scores.

        :param data_i: A string for which sentiment analysis is performed.
        :return: A dictionary with sentiment scores.
        """
        return dict(self.sentiment_model(data_i)._.polarity)

    def get_aggregate_sentiment(self, data):
        """
        Computes aggregate sentiment for a collection of texts.
        - Determines overall sentiment and the most significant positive and negative cases.
        - Generates a visualization of the most significant sentiment.

        :param data: A list of strings to analyze.
        :return: A dictionary containing aggregate sentiment information.
        """
        data = data.tolist()
        negsum, possum = 0, 0
        most_negative = (None, float('-inf'))
        most_positive = (None, float('-inf'))
        for d in data:
            doc = self.get_sentiment(d)
            negsum += doc['negative']
            possum += doc['positive']
            if abs(doc['negative']) > most_negative[1]:
                most_negative = (doc, abs(doc['negative']))
            if abs(doc['positive']) > most_positive[1]:
                most_positive = (doc, abs(doc['positive']))
        res = {'positive': possum, 'negative': negsum}
        delta = max(possum, negsum) - min(possum, negsum)
        # Find the key with the maximum value
        max_key = max(res, key=res.get)
        max_value = res[max_key]

        if max_key == 'negative':
            svg_image = asent.visualize(most_negative[0], style='prediction')
        else:
            svg_image = asent.visualize(most_positive[0], style='prediction')
        return {'sentiment': max_key, 'sentiment_score': max_value, 'delta': delta, 'svg_image': svg_image}

    def build_embedding_matrix(self):
        """
        Builds an embedding matrix for words in news titles.
        - Uses NLP model to get word embeddings.
        - Applies t-SNE for dimensionality reduction.
        - Categorizes words based on their occurrence in different news sources.
        """
        # get most commonly used words in english language to serve as baseline cluster
        most_common_100 = set(pd.read_csv('mostcommonwords.csv').iloc[:, 0].tolist())
        # Extract words from titles
        leftwing_words = [word for sentence in self.leftwing_titles for word in sentence.split() if word.isalpha()]
        rightwing_words = [word for sentence in self.rightwing_titles for word in sentence.split() if word.isalpha()]
        # Calculate intersection and unique words (found in articles returned by query)
        intersection = set(leftwing_words) & set(rightwing_words)
        strictly_leftwing_words = set(leftwing_words) - intersection
        strictly_rightwing_words = set(rightwing_words) - intersection
        # Combine all unique words
        all_words = list(intersection | strictly_leftwing_words | strictly_rightwing_words | most_common_100)
        # Get embeddings
        docs = [self.nlp_model(word) for word in all_words]
        # Dimensionality reduction with t-SNE
        tsne = TSNE(n_components=3, random_state=0)
        embeddings = np.array([word.vector for word in docs])
        reduced_embeddings = tsne.fit_transform(embeddings)
        reduced_embeddings = (reduced_embeddings - reduced_embeddings.min(axis=0)) / (reduced_embeddings.max(axis=0) - reduced_embeddings.min(axis=0))
        x = reduced_embeddings[:, 0].reshape(-1)
        y = reduced_embeddings[:, 1].reshape(-1)
        z = reduced_embeddings[:, 2].reshape(-1)
        labels = ['intersection' if (word in intersection or word in most_common_100) else
                  'left-wing' if (word in strictly_leftwing_words) else
                  'right-wing' for word in all_words]

        self.embedding_projection_data =  pd.DataFrame({'word': all_words,
                             'x': x,
                             'y': y,
                             'z': z,
                             'partisan': labels,
                             })








