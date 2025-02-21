import os
import requests
import streamlit as st
from dotenv import load_dotenv
from transformers import pipeline
from langchain_community.llms import HuggingFaceHub
from langchain import PromptTemplate, LLMChain

# Load API Token
load_dotenv()
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Validate Image URL
def is_valid_image_url(url):
    try:
        response = requests.get(url, stream=True)
        return response.status_code == 200 and response.headers.get("Content-Type", "").startswith("image")
    except requests.RequestException:
        return False

# Image to Text Function
def img2text_url(image_url):
    API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}

    response = requests.post(API_URL, headers=headers, json={"inputs": image_url})
    if response.status_code == 200:
        return response.json()[0].get("generated_text", "No caption generated.")
    return f"Error: {response.status_code}, {response.text}"

# Generate Story Function with Fallback Model
# Generate Story Function with Fallback Model
def generate_story(scenario):
    try:
        template = """
        You are a story teller;
        Generate a short, creative story (10-50 words) based on the given narrative.
        CONTEXT: {scenario}
        STORY:
        """
        prompt = PromptTemplate(template=template, input_variables=["scenario"])

        # Primary Model
        try:
            story_llm = LLMChain(llm=HuggingFaceHub(
                repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                model_kwargs={"temperature": 1, "max_length": 1024}
            ), prompt=prompt)
        except Exception:
            story_llm = LLMChain(llm=HuggingFaceHub(
                repo_id="tiiuae/falcon-7b-instruct",
                model_kwargs={"temperature": 1, "max_length": 1024}
            ), prompt=prompt)

        # Extract only the story part
        full_response = story_llm.predict(scenario=scenario).strip()
        story_part = full_response.split("\n")[-1].strip()  # Get only the generated story

        return story_part
    except Exception as e:
        return f"Error generating story: {str(e)}"


# Text to Speech Function
def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
    payload = {'inputs': message}

    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        audio_path = "audio.flac"
        with open(audio_path, "wb") as file:
            file.write(response.content)
        return audio_path
    return f"Error: {response.status_code}, {response.text}"

# Streamlit UI
def main():
    st.set_page_config(page_title="AI-Powered Story Generator", layout="centered")
    st.title("📖 AI Story Generator from Image URL")

    image_url = st.text_input("🔗 Enter Image URL")

    if image_url:
        if not is_valid_image_url(image_url):
            st.error("❌ Invalid image URL. Please enter a valid image link.")
            return

        st.image(image_url, caption="Uploaded Image", use_container_width=True)
        caption = img2text_url(image_url)
        if caption.startswith("Error"):
            st.error(f"❌ Failed to generate caption: {caption}")
            return
        st.success("✅ Image caption generated successfully!")

        story = generate_story(caption)
        if story.startswith("Error"):
            st.error(f"❌ Failed to generate story: {story}")
            return
        st.success("✅ Story generated successfully!")

        audio_file = text2speech(story)
        if audio_file.startswith("Error"):
            st.error(f"❌ Failed to generate audio: {audio_file}")
            return
        st.success("✅ Audio generated successfully!")

        with st.expander("📷 See Image Caption"):
            st.write(caption)

        with st.expander("📖 Read Generated Story"):
            st.write(story)

        with st.expander("🔊 Listen to the Story"):
            st.audio(audio_file, format="audio/flac")

if __name__ == "__main__":
    main()
