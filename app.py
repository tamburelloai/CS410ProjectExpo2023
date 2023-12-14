import subprocess
import streamlit as st
import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from backend import Backend
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
from utils import *

# PAGE CONFIGURATION
st.set_page_config(layout='wide')
#set_custom_css()

if 'input' not in st.session_state:
    st.session_state.input = True



# MAIN APPLICATION
st.title("Visualized Media Bias and Polarization Detection")
if st.session_state.input:
    # Show input widgets if in input mode
    st.write("Utilizes cutting-edge Natural Language Processing (NLP) techniques to analyze and visualize media bias across various news outlets. The platform's primary function is to identify and quantify ideological biases and narrative divergences in news reporting, using advanced tools like spaCy and PyTorch. Users will be able to see dynamic visualizations that highlight differences in media language and portrayal of current events, offering a clearer understanding of media polarization. The platform aims to enhance public dialogue and support democratic values by making users aware of the biases in news sources.")
    st.button('Get Started', on_click=show_result) # Callback changes it to result mode
else:
    selected_page = "Home"
    with st.sidebar:
        st.title("Navigation")
        if st.button("&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Home &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; "):
            selected_page = "Home"
            st.write("You are on the Home page.")
        st.write("---")  # Optional: adds a visual separator
        st.title("Documentation")
        if st.button("&nbsp; &nbsp; &nbsp; &nbsp; Backend Docs &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;"):
            selected_page = "Backend Documentation"
        if st.button("NewsApiManager Docs"):
            selected_page = "NewsApiManager Documentation"
        if st.button("&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Utilities Docs &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;"):
            selected_page = "Utilities Documentation"
    page = selected_page
    # Load and display the HTML file when "HTML Page" is selected
    if page == "Backend Documentation":
        html_file_path = 'html/backend.html'
        # Read the HTML file
        with open(html_file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        # Render the HTML file
        components.html(html_content, height=600, scrolling=True)

    elif page == "NewsApiManager Documentation":
        html_file_path = 'html/news_api_manager.html'
        # Read the HTML file
        with open(html_file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        # Render the HTML file
        components.html(html_content, height=600, scrolling=True)

    elif page == "Utilities Documentation":
        html_file_path = 'html/utils.html'
        # Read the HTML file
        with open(html_file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        # Render the HTML file
        components.html(html_content, height=600, scrolling=True)
    else:
        st.subheader("Enter a query")
        query_string = st.text_input("")

        backend = Backend()
        if query_string:
            res = backend.run_query(query_string)
            if not res:
                st.write("No responses found. Please try another query")
            else:
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








