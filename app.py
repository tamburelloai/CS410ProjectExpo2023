import subprocess
import streamlit as st
@st.cache_resource
def download_en_core_web_md():
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_md"])

download_en_core_web_md()
import datetime
import plotly.express as px
#TABLE
import plotly.graph_objects as go
import pandas as pd
from backend import Backend
import streamlit as st
st.set_page_config(layout="wide")
import numpy as np

background_color = "#111111"  # Example background color: white
navbar_color = "#0000FF"  # Example navbar color: blue
text_color = "rgba(255, 0, 0, 255)"  # Example text color: black
header_color = "rgba(255, 235, 122, 0.85)"  # Example subheader color: green
subheader_color = "rgba(255, 235, 122, 0.7)"  # Example subheader color: green
text_color = "rgba(255, 235, 122, 0.55)"  # Example subheader color: green


def set_custom_css():

    css = f"""
        <style>
        /* Set the sidebar color */
        .stSidebar {{
            background-color: {background_color};
        }}

        /* Set the top navigation bar color */
        header .css-1aumxhk {{
            background-color: {navbar_color};
        }}

        /* Set the background color of the main content area */
        .stApp {{
            background-color: {background_color};
            color: {text_color};
        }}

        /* Set the header color */
        h1 {{ 
            color: {header_color};  /* Example: Red color for main header */
        }}

        /* Set the subheader color */
        h2 {{
            color: {subheader_color};  /* Example: Green color for subheader */
        }}
        
        h3 {{
            color: {subheader_color};  /* Example: Green color for subheader */
        }}
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)

    def hide_img_fs():
        st.markdown('''
        <style>
        button[title="View fullscreen"]{
            visibility: hidden;}
        </style>
        ''', unsafe_allow_html=True)



def show_result():
    st.session_state.input = False

def reset():
    st.session_state.input = True

def plot_projection_visualization(df):
    color_mapping = {
        'intersection': 'rgba(0.5, 0.5, 0.5, 0.5)',
        'left-wing': 'rgba(0.0, 0.0, 0.5, 0.5)',
        'right-wing': 'rgba(0.5, 0.0, 0.0, 0.5)'
    }

    opacity_mapping = {
        'intersection': 0.7,
        'left-wing': 0.7,
        'right-wing': 0.7
    }




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
            color=df['partisan'].map(color_mapping)
        ),
        text=df['word'],
        hoverinfo='text',
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
            xaxis=dict(showbackground=False, backgroundcolor=header_color,
                       gridcolor=header_color, showticklabels=False),
            yaxis=dict(showbackground=False, backgroundcolor=header_color,
                       gridcolor=header_color, showticklabels=False),
            zaxis=dict(showbackground=False, backgroundcolor=header_color,
                       gridcolor=header_color, showticklabels=False),
            camera=dict(
                eye=dict(x=1.25, y=1.25, z=1.25)
            )
        ),
        paper_bgcolor="rgb(255, 235, 122)",
        plot_bgcolor='rgb(255, 235, 122)',
        margin=dict(l=0, r=0, b=0, t=0)
    )

    # Optionally, remove the legend if it's not needed
    fig.update_layout(showlegend=False)
    fig.update_layout(autosize=True)
    st.plotly_chart(fig, use_container_width=True)

def render_table(df, wing):
    def color_cells_by_sentiment(row):
        """
        Colors the cells based on the 'sentiment' value.
        Red for negative sentiment, green for positive.
        The intensity of the color is proportional to the absolute sentiment value.
        """
        sentiment_value = row['sentiment']
        base_color_red = 255
        base_color_green = 255

        if sentiment_value < 0:
            color_intensity = min(int(255 + (255 * sentiment_value)), 255)
            color = f'rgb({base_color_red}, {color_intensity}, {color_intensity})'  # More red for negative
        else:
            color_intensity = min(int(255 - (255 * sentiment_value)), 255)
            color = f'rgb({color_intensity}, {base_color_green}, {color_intensity})'  # More green for positive

        return ['background-color: %s' % color for _ in row]

    styled_df = df.style.apply(color_cells_by_sentiment, axis=1)
    st.dataframe(styled_df, hide_index=True)

def render_average_sentiment(df, wing):
    avg_sentiment = np.mean(df.sentiment.values)
    if avg_sentiment > 0.0:
        st.write(f'{wing} sentiment: positive'.upper())
    elif avg_sentiment < 0.0:
        st.write(f'{wing} sentiment: negative'.upper())
    else:
        st.write(f'{wing} sentiment: positive'.upper())
    st.metric('average sentiment', np.round(avg_sentiment, 3))


def render_sentiment_distribution(left_df, right_df):
    # Calculate mean sentiments
    mean_left_sentiment = left_df['sentiment'].mean()
    mean_right_sentiment = right_df['sentiment'].mean()

    # Combine dataframes with an additional column to distinguish them
    left_df['Source'] = 'Left'
    right_df['Source'] = 'Right'
    combined_df = pd.concat([left_df, right_df])
    # Define colors for each source
    color_discrete_map = {'Left': 'rgba(0.0, 0.0, 1.0, 0.85)',
                          'Right': 'rgba(1.0, 0.0, 0.0, 0.85)'}

    # Create histogram
    fig = px.histogram(combined_df, x='sentiment', color='Source', barmode='overlay', nbins=10,
                       color_discrete_map=color_discrete_map, height=250)

    # Add mean lines
    fig.add_vline(x=mean_left_sentiment, line_width=3, line_dash="dot", line_color="blue", annotation_text="Left Mean")
    fig.add_vline(x=mean_right_sentiment, line_width=3, line_dash="dot", line_color="red", annotation_text="Right Mean")
    fig.add_vline(x=0, line_width=3, line_dash="solid", line_color="black", annotation_text="Right Mean")
    # Update layout
    fig.update_layout(
        title='Distribution of Sentiment Scores',
        xaxis_title='Sentiment Score',
        yaxis_title='Count',
        paper_bgcolor=header_color,
    )
    # Display plot
    st.plotly_chart(fig, use_container_width=True)


set_custom_css()

if 'input' not in st.session_state:
    st.session_state.input = True



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








