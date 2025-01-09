import os
import pandas as pd
import numpy as np
import streamlit as st 
import google.generativeai as genai
import time
import tempfile
from pathlib import Path
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file, get_file

# Setting up the Page Configuration
st.set_page_config(
    page_title="AI Analyst",
    page_icon="🧠",
    layout="wide"
)

st.title("AI InsightHub")
st.header("Leverage Advanced AI Insights from Videos, Data, and Images")
st.caption(
    'This AI-driven Streamlit application harnesses the power of advanced machine learning algorithms to provide in-depth analysis across various file types, including videos, data, and images. With cutting-edge AI techniques, users can automatically extract valuable information, detect patterns, and gain actionable insights. Whether analyzing large datasets, leveraging computer vision for image and video analysis, or utilizing AI-based predictive models, this app offers a seamless, intelligent solution for unlocking the full potential of your files.'
)

# User-provided API Key
st.sidebar.subheader("API Configuration")
api_key = st.sidebar.text_input(
    "Enter your Google API Key",
    type="password",
    help="Your API key will be used to configure the application for analysis."
)

if api_key:
    try:
        genai.configure(api_key=api_key)
        st.sidebar.success("API key validated successfully!")
    except Exception as e:
        st.sidebar.error(f"Invalid API key: {e}")
        st.stop()
else:
    st.sidebar.warning("Please enter a valid API key to proceed.")
    st.stop()

@st.cache_resource
def initialize_agent():
    return Agent(
        name="Media AI Analyzer",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo()],
        markdown=True,
    )

# Custom CSS
st.markdown("""
    <style>
    .main { padding: 0rem 1rem; }
    .stTextArea textarea { height: 100px; }
    .analysis-box { background-color: #2b2b2b; padding: 20px; border-radius: 10px; margin: 10px 0; }
    .sidebar-content { padding: 20px; margin-bottom: 20px; }
    .title-text { color: #1E88E5; font-size: 24px; font-weight: bold; }
    .subtitle-text { color: #666; font-size: 18px; }
    </style>
""", unsafe_allow_html=True)

## Initialize the agent
multimodal_Agent = initialize_agent()

# Create tabs for different media types
media_type = st.tabs(["Video Analysis", "Image Analysis", "Data Analysis"])

with media_type[0]:  # Video Analysis Tab
    st.subheader("Video Analysis")
    video_file = st.file_uploader("Upload a video file", type=['mp4', 'mov', 'avi'], key="video_uploader", 
                                 help="Upload a video for AI analysis")

    if video_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            temp_video.write(video_file.read())
            video_path = temp_video.name

        st.video(video_path, format="video/mp4", start_time=0)

        user_query = st.text_area(
            "What insights are you seeking from the video?",
            placeholder="Ask anything about the video content. The AI agent will analyze and gather additional context if needed.",
            help="Provide specific questions or insights you want from the video.",
            key="video_query"
        )

        if st.button("🔍 Analyze Video", key="analyze_video_button"):
            if not user_query:
                st.warning("Please enter a question or insight to analyze the video.")
            else:
                try:
                    with st.spinner("Processing video and gathering insights..."):
                        # Upload and process video file
                        processed_video = upload_file(video_path)
                        while processed_video.state.name == "PROCESSING":
                            time.sleep(1)
                            processed_video = get_file(processed_video.name)

                        # Prompt generation for video analysis
                        analysis_prompt = f"""
                            Analyze the uploaded video for content and context.
                            Respond to the following query using video insights and supplementary web research:
                            {user_query}

                            Provide a detailed, user-friendly, and actionable response.
                            """

                        # AI agent processing
                        response = multimodal_Agent.run(analysis_prompt, videos=[processed_video])

                    # Display the result
                    st.subheader("Analysis Result")
                    st.markdown(response.content)

                except Exception as error:
                    st.error(f"An error occurred during analysis: {error}")
                finally:
                    # Clean up temporary video file
                    Path(video_path).unlink(missing_ok=True)

with media_type[1]:  # Image Analysis Tab
    st.subheader("Image Analysis")
    image_file = st.file_uploader("Upload an image file", type=['png', 'jpg', 'jpeg'], key="image_uploader",
                                 help="Upload an image for AI analysis")

    if image_file:
        # Display the uploaded image
        st.image(image_file, caption="Uploaded Image", use_container_width=True)

        user_query = st.text_area(
            "What would you like to know about this image?",
            placeholder="Ask anything about the image content. The AI agent will analyze and provide detailed insights.",
            help="Provide specific questions or aspects you want analyzed in the image.",
            key="image_query"
        )

        if st.button("🔍 Analyze Image", key="analyze_image_button"):
            if not user_query:
                st.warning("Please enter a question or insight to analyze the image.")
            else:
                try:
                    with st.spinner("Analyzing image and gathering insights..."):
                        # Create temporary file for image
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_image:
                            temp_image.write(image_file.getbuffer())
                            image_path = temp_image.name

                        # Upload and process image file
                        processed_image = upload_file(image_path)

                        # Prompt generation for image analysis
                        analysis_prompt = f"""
                            Analyze the uploaded image in detail.
                            Respond to the following query using image analysis and supplementary web research:
                            {user_query}

                            Provide a comprehensive, detailed, and informative response covering visual elements, 
                            context, and any relevant insights.
                            """

                        # AI agent processing
                        response = multimodal_Agent.run(analysis_prompt, images=[processed_image])

                    # Display the result
                    st.subheader("Analysis Result")
                    st.markdown(response.content)

                except Exception as error:
                    st.error(f"An error occurred during analysis: {error}")
                finally:
                    # Clean up temporary image file
                    Path(image_path).unlink(missing_ok=True)

with media_type[2]:  # Data Analysis Tab
    st.subheader("Data Analysis")
    data_file = st.file_uploader("Upload a data file (Excel or CSV)", type=['csv', 'xlsx'], key="data_uploader",
                                 help="Upload a dataset for AI analysis")

    if data_file:
        try:
            # Read and parse the uploaded file
            if data_file.name.endswith('.csv'):
                data = pd.read_csv(data_file)
            elif data_file.name.endswith('.xlsx'):
                data = pd.read_excel(data_file)

            # Display the uploaded data
            st.write("Uploaded Dataset:")
            st.dataframe(data.head())

            # Serialize the dataset for AI analysis
            dataset_text = data.to_string(index=False)

            # User query input for insights
            user_query = st.text_area(
                "What would you like to know about this dataset?",
                placeholder="Ask anything about the dataset. The AI agent will analyze and provide detailed insights.",
                help="Provide specific questions or ask for general insights about the data.",
                key="data_query"
            )

            if st.button("🔍 Analyze Data", key="analyze_data_button"):
                if not user_query:
                    st.warning("Please enter a question or insight to analyze the data.")
                else:
                    try:
                        with st.spinner("Analyzing data and gathering insights..."):
                            # Prepare the analysis prompt
                            analysis_prompt = f"""
                                Here is the dataset:
                                {dataset_text}

                                Respond to the following query using data analysis techniques:
                                {user_query}

                                Provide relevant statistics, trends, or actionable insights based on the dataset.
                            """

                            # AI agent processing
                            response = multimodal_Agent.run(analysis_prompt)

                        # Display the result
                        st.subheader("Analysis Result")
                        st.markdown(response.content)

                    except Exception as error:
                        st.error(f"An error occurred during analysis: {error}")

        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")

# Customize text area height
st.markdown(
    """
    <style>
    .stTextArea textarea {
        height: 100px;
    }
    </style>
    """,
    unsafe_allow_html=True
)