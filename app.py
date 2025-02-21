import os
import requests
import streamlit as st
from transformers import pipeline
from langchain_community.llms import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
from dotenv import load_dotenv
from gtts import gTTS

# Load environment variables
load_dotenv()

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
    headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACEHUB_API_TOKEN')}"}

    response = requests.post(API_URL, headers=headers, json={"inputs": image_url})
    if response.status_code == 200:
        return response.json()[0].get("generated_text", "No caption generated.")
    return f"Error: {response.status_code}, {response.text}"

# Generate Story Function
def generate_story(scenario):
    template = """
    You are a story teller;
    You can generate a very short story based on a simple narrative, be creative and the story should be between 10 to 50 words;
    CONTEXT: {scenario}
    STORY:
    """
    prompt = PromptTemplate(template=template, input_variables=["scenario"])
    story_llm = LLMChain(
        llm=HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct", model_kwargs={"temperature": 1, "max_length": 512}),
        prompt=prompt
    )
    return story_llm.predict(scenario=scenario).strip()

# Text to Speech Function (Using gTTS)
def text2speech(message):
    try:
        tts = gTTS(text=message, lang="en")
        audio_path = "audio.mp3"
        tts.save(audio_path)
        return audio_path
    except Exception as e:
        return f"Error: {str(e)}"

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
            st.audio(audio_file, format="audio/mp3")

if __name__ == "__main__":
    main()
