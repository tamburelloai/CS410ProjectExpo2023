import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit as st
import numpy as np


# CONSTANTS
background_color = "#111111"
navbar_color = "#0000FF"  # Example navbar color: blue
text_color = "rgba(255, 0, 0, 255)"  # Example text color: black
header_color = "rgba(255, 235, 122, 0.85)"  # Example subheader color: green
subheader_color = "rgba(255, 235, 122, 0.7)"  # Example subheader color: green
text_color = "rgba(255, 235, 122, 0.55)"  # Example subheader color: green
left_wing_sources = [
    "the-huffington-post",
    "msnbc",
    "cnn",
    "the-washington-post",
    "buzzfeed",
    "the-guardian",
]
right_wing_sources = [
    "fox-news",
    "breitbart-news",
    "the-washington-times",
    "the-american-conservative",
    "newsmax",
    "the-federalist",
]


def set_custom_css():
    """
    Sets custom CSS styles for the Streamlit app.
    - Defines and applies styles for sidebar, navbar, headers, subheaders, etc.
    """
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
        
        .stButton>button {{
            color: white;
            background-color: #0d6efd; /* Bootstrap primary color */
            border-color: #0d6efd;
            border-radius: 5px; /* Optional: for rounded corners */
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


def show_result():
    """
        Toggles the Streamlit session state to show results instead of input fields.
        - Used in button callbacks to switch the app's mode.
        """
    st.session_state.input = False


def reset():
    """
        Resets the Streamlit session state to show input fields again.
        - Used to reset the app to its initial state.
        """
    st.session_state.input = True


def plot_projection_visualization(df):
    """
        Plots a 3D projection visualization using Plotly.
        - Visualizes data points in 3D space with X, Y, and Z axes.
        - Color codes and labels the points based on their attributes.

        :param df: A pandas DataFrame containing x, y, z coordinates and labels for the data points.
        """
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
            xaxis=dict(showbackground=False, showline=False, zeroline=False, showticklabels=False, title='',
                       showspikes=False),
            yaxis=dict(showbackground=False, showline=False, zeroline=False, showticklabels=False, title='',
                       showspikes=False),
            zaxis=dict(showbackground=False, showline=False, zeroline=False, showticklabels=False, title='',
                       showspikes=False),
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
    """
       Renders a styled table in Streamlit.
       - Colors table cells based on sentiment values.
       - Displays the DataFrame in a user-friendly format.

       :param df: A pandas DataFrame containing data to display in the table.
       :param wing: A string indicating the political wing (e.g., 'leftwing', 'rightwing') for labeling.
       """

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
    """
       Renders the average sentiment for a given wing.
       - Calculates and displays the average sentiment score.
       - Indicates whether the overall sentiment is positive, negative, or neutral.

       :param df: A pandas DataFrame containing sentiment data.
       :param wing: A string indicating the political wing for labeling purposes.
       """
    avg_sentiment = np.mean(df.sentiment.values)
    if avg_sentiment > 0.0:
        st.write(f'{wing} sentiment: positive'.upper())
    elif avg_sentiment < 0.0:
        st.write(f'{wing} sentiment: negative'.upper())
    else:
        st.write(f'{wing} sentiment: positive'.upper())
    st.metric('average sentiment', np.round(avg_sentiment, 3))


def render_sentiment_distribution(left_df, right_df):
    """
        Renders a sentiment distribution chart for left and right wings.
        - Creates a histogram to compare sentiment distributions.
        - Displays mean lines and labels for each wing.

        :param left_df: A pandas DataFrame containing sentiment data for the left wing.
        :param right_df: A pandas DataFrame containing sentiment data for the right wing.
        """
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

