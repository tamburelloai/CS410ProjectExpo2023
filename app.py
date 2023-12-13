import subprocess
import streamlit as st
import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from backend import Backend
import streamlit as st
import numpy as np
from utils import *


# PAGE CONFIGURATION
st.set_page_config(layout='wide')
set_custom_css()
if 'input' not in st.session_state:
    st.session_state.input = True

# MAIN APPLICATION
st.title("Visualized Media Bias and Polarization Detection")
if st.session_state.input:
    # Show input widgets if in input mode
    st.write("Utilizes cutting-edge Natural Language Processing (NLP) techniques to analyze and visualize media bias across various news outlets. The platform's primary function is to identify and quantify ideological biases and narrative divergences in news reporting, using advanced tools like spaCy and PyTorch. Users will be able to see dynamic visualizations that highlight differences in media language and portrayal of current events, offering a clearer understanding of media polarization. The platform aims to enhance public dialogue and support democratic values by making users aware of the biases in news sources. The app's effectiveness will be evaluated through precision, recall, F1-score, ROC-AUC metrics, and user feedback.")
    st.button('Get Started', on_click=show_result) # Callback changes it to result mode
else:

    st.subheader("Enter a query")
    query_string = st.text_input("")

    backend = Backend()
    if query_string:
        backend.run_query(query_string)
        print(f'THERE IS A STRING AT  {datetime.datetime.now()}')

        st.divider()
        st.markdown(backend.get_entity_breakdown_html(query_string), unsafe_allow_html=True)
        st.divider()

        backend.build_embedding_matrix()
        render_sentiment_distribution(left_df=backend.leftwing_dataframe, right_df=backend.rightwing_dataframe)
        plot_projection_visualization(backend.embedding_projection_data)
        col1, col2 = st.columns(2)

        # Left table
        with col1:
            render_average_sentiment(backend.leftwing_dataframe, wing='leftwing')
            render_table(backend.leftwing_dataframe, wing='leftwing')

        # Right table
        with col2:
            render_average_sentiment(backend.rightwing_dataframe, wing='rightwing')
            render_table(backend.rightwing_dataframe, wing='rightwing')








