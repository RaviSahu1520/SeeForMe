import streamlit as st
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.messages import HumanMessage
import pytesseract
from gtts import gTTS
import io
import base64
import logging

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure Google API Key
api_key = st.secrets["google_genai"]["api_key"]

# Initialize models with correct API key
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
vision_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

# Error handling function
def handle_error(error):
    logging.error(error)
    st.error(f"Error: {str(error)}")

# Scene understanding function
def scene_understanding(image):
    try:
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')
        image_bytes = image_bytes.getvalue()

        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": """As an AI assistant for visually impaired individuals, provide a detailed description of this image.
                    Include:
                    1. Overall scene layout
                    2. Main objects and their positions
                    3. People and their activities (if any)
                    4. Colors and lighting
                    5. Notable features or points of interest

                    Format the response in clear, easy-to-understand sections."""
                },
                {
                    "type": "image_url",
                    "image_url": f"data:image/png;base64,{base64.b64encode(image_bytes).decode()}"
                }
            ]
        )

        response = vision_llm.invoke([message])
        return response.content
    except Exception as e:
        handle_error(e)

# Object detection function
def detect_objects_and_obstacles(image):
    try:
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')
        image_bytes = image_bytes.getvalue()

        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": """Analyze this image for safety and navigation purposes.
                    Please provide:
                    1. List of potential obstacles or hazards
                    2. Safe paths for navigation
                    3. Approximate distances and spatial relationships
                    4. Safety warnings if any
                    5. Guidance for moving through the space

                    Format the response in clear sections with bullet points where appropriate."""
                },
                {
                    "type": "image_url",
                    "image_url": f"data:image/png;base64,{base64.b64encode(image_bytes).decode()}"
                }
            ]
        )

        response = vision_llm.invoke([message])
        return response.content
    except Exception as e:
        handle_error(e)

# Text extraction function
def extract_and_process_text(image):
    try:
        extracted_text = pytesseract.image_to_string(image)

        if not extracted_text.strip():
            return "No text detected in the image."

        template = """
        Enhance and structure the following extracted text for a visually impaired person:
        TEXT: {text}

        Please:
        1. Correct obvious OCR errors
        2. Format text in clear sections
        3. Highlight important information
        4. Add relevant context
        5. Organize numbers, dates, and key details

        Return the text in a clear, well-structured format.
        """

        prompt = PromptTemplate(input_variables=["text"], template=template)
        chain = LLMChain(llm=llm, prompt=prompt)

        return chain.run(text=extracted_text)
    except Exception as e:
        handle_error(e)

# Task assistance function
def provide_task_assistance(image, task_type):
    try:
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')
        image_bytes = image_bytes.getvalue()

        task_prompts = {
            "item_identification": """
                Identify and describe items in this image:
                1. Name and type of each item
                2. Key features and characteristics
                3. Important details for identification
                4. Relative positions and arrangements
                5. Any warning labels or important markings
            """,
            "label_reading": """
                Analyze labels and text in this image:
                1. Brand names and product types
                2. Important warnings or instructions
                3. Expiration dates or crucial numbers
                4. Ingredient lists or contents
                5. Usage instructions if visible
            """,
            "navigation_help": """
                Provide navigation guidance for this space:
                1. Key landmarks and reference points
                2. Potential obstacles or hazards
                3. Suggested path or direction
                4. Distance estimates
                5. Safety considerations
            """,
            "daily_tasks": """
                Provide guidance for daily tasks in this scene:
                1. Step-by-step instructions
                2. Safety precautions
                3. Important object locations
                4. Helpful tips and suggestions
                5. Potential challenges to be aware of
            """
        }

        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": task_prompts.get(task_type, task_prompts["item_identification"])
                },
                {
                    "type": "image_url",
                    "image_url": f"data:image/png;base64,{base64.b64encode(image_bytes).decode()}"
                }
            ]
        )

        response = vision_llm.invoke([message])
        return response.content
    except Exception as e:
        handle_error(e)

# Text-to-speech function
def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='en')
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        return mp3_fp.getvalue()
    except Exception as e:
        handle_error(e)

# Streamlit app main
def main():
    st.set_page_config(page_title="Vision Assistant", layout="wide")

    st.title("SeeForMe : AI Assistant for Visually Impaired")

    st.sidebar.title("About SeeForMe")
    st.sidebar.info(
        "AI Assistant for Visually Impaired\n"
        "• Upload any image to:\n"
        "• Get detailed scene descriptions\n"
        "• Read text from images\n"
        "• Detect objects & obstacles\n"
        "• Receive task-specific guidance"
    )

    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])

        with col1:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

        with col2:
            feature = st.radio(
                "Select Feature",
                ["Scene Description", "Text Reading", "Object Detection", "Task Assistance"],
                help="Choose the type of assistance you need"
            )

            if feature == "Scene Description":
                if st.button("Analyze Scene", use_container_width=True):
                    with st.spinner("Analyzing the scene..."):
                        description = scene_understanding(image)
                        st.write(description)
                        audio_bytes = text_to_speech(description)
                        st.audio(audio_bytes, format='audio/mp3')

            elif feature == "Text Reading":
                if st.button("Extract & Read Text", use_container_width=True):
                    with st.spinner("Processing text..."):
                        text_content = extract_and_process_text(image)
                        st.write(text_content)
                        audio_bytes = text_to_speech(text_content)
                        st.audio(audio_bytes, format='audio/mp3')

            elif feature == "Object Detection":
                if st.button("Detect Objects", use_container_width=True):
                    with st.spinner("Analyzing objects and obstacles..."):
                        objects_info = detect_objects_and_obstacles(image)
                        st.write(objects_info)
                        audio_bytes = text_to_speech(objects_info)
                        st.audio(audio_bytes, format='audio/mp3')

            else:  # Task Assistance
                task_type = st.selectbox(
                    "Select Task Type",
                    ["item_identification", "label_reading", "navigation_help", "daily_tasks"],
                    format_func=lambda x: x.replace('_', ' ').title()
                )

                if st.button("Get Assistance", use_container_width=True):
                    with st.spinner("Generating guidance..."):
                        guidance = provide_task_assistance(image, task_type)
                        st.write(guidance)
                        audio_bytes = text_to_speech(guidance)
                        st.audio(audio_bytes, format='audio/mp3')

if __name__ == "__main__":
    main()
