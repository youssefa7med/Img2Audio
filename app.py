import requests
import streamlit as st
import os
from transformers import pipeline
from langchain.llms import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
from dotenv import load_dotenv

load_dotenv()


def is_valid_image_url(url):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200 and response.headers["Content-Type"].startswith("image"):
            return True
    except Exception:
        return False
    return False

# Image to Text Function
def img2text_url(image_url):
    API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
    headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACEHUB_API_TOKEN')}"}
    
    response = requests.post(API_URL, headers=headers, json={"inputs": image_url})
    
    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        return f"Error: {response.status_code}, {response.json()}"

# Generate Story Function
def generate_story(scenario):
    template = """
    You are a story teller;
    You can generate a very short story based on a simple narrative, be creative and the story should be between 10 to 50 words;
    CONTEXT: {scenario}
    STORY:
    """
    prompt = PromptTemplate(template=template, input_variables=["scenario"])
    
    try:
        story_llm = LLMChain(
            llm=HuggingFaceHub(repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 
                               model_kwargs={"temperature":1, "max_length":512}), 
            prompt=prompt
        )
        story = story_llm.predict(scenario=scenario)
    except Exception as e:
        print(f"DeepSeek model failed: {e}, switching to Falcon-7B-Instruct...")
        story_llm = LLMChain(
            llm=HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", 
                               model_kwargs={"temperature":1, "max_length":512}), 
            prompt=prompt
        )
        story = story_llm.predict(scenario=scenario)
    
    return story.split('\n')[-1].strip()

# Text to Speech Function
def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACEHUB_API_TOKEN')}"}
    payload = {'inputs': message}
    
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        with open("audio.flac", "wb") as file:
            file.write(response.content)
        return "audio.flac"
    else:
        return None

# Streamlit UI
def main():
    st.set_page_config(page_title="AI-Powered Story Generator", layout="centered")
    st.title("AI Story Generator from Image URL")
    
    image_url = st.text_input("Enter Image URL")
    
    if image_url:
        if is_valid_image_url(image_url):
            st.image(image_url, caption="Uploaded Image", use_container_width =True)
            
            progress = st.progress(0)
            progress.progress(25)
            caption = img2text_url(image_url)
            st.success("Image caption generated successfully!")
            progress.progress(50)
            
            story = generate_story(caption)
            st.success("Story generated successfully!")
            progress.progress(75)
            
            audio_file = text2speech(story)
            st.success("Audio generated successfully!")
            progress.progress(100)
            
            with st.expander("See Image Caption"):
                st.write(caption)
            
            with st.expander("Read Generated Story"):
                st.write(story)
            
            if audio_file:
                with st.expander("Listen to the Story"):
                    st.audio(audio_file, format="audio/flac")
        else:
            st.error("Invalid image URL. Please enter a valid image link.")

if __name__ == "__main__":
    main()
