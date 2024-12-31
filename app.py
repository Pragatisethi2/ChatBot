import os
import base64
import io
import sqlite3

import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from datetime import datetime
from openai import OpenAI

# ---------------------- Load Environment ----------------------
load_dotenv()  # Loads .env file if present
API_KEY = st.secrets["OPENAI_API_KEY"]

if not API_KEY:
    st.error("OpenAI API key not found. Please set OPENAI_API_KEY in .env or environment variables.")
    st.stop()

# ---------------------- Initialize OpenAI Client ----------------------
def load_model():
    """
    Create an OpenAI client instance.
    """
    return OpenAI(api_key=API_KEY)

# ---------------------- Setup SQLite Database ----------------------
conn = sqlite3.connect("conversations.db", check_same_thread=False)
c = conn.cursor()

# Create a table to store user prompt, image (base64), response, timestamp
c.execute(
    """
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_prompt TEXT,
        base64_image TEXT,
        bot_response TEXT,
        timestamp TEXT
    )
    """
)
conn.commit()

def save_conversation(user_prompt, base64_image, bot_response):
    """
    Inserts a record of the user prompt, optional image, bot response, and timestamp.
    """
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute(
        "INSERT INTO conversations (user_prompt, base64_image, bot_response, timestamp) VALUES (?, ?, ?, ?)",
        (user_prompt, base64_image, bot_response, timestamp_str),
    )
    conn.commit()

def load_conversations():
    """
    Loads all conversations from the database, most recent first.
    """
    c.execute("SELECT user_prompt, base64_image, bot_response, timestamp FROM conversations ORDER BY id DESC")
    return c.fetchall()

# ---------------------- Helper: Encode Image to Base64 ----------------------
def encode_image_to_base64(pil_image):
    """
    Encodes a PIL image to base64 format.
    """
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

# ---------------------- Analyze (Text + Image) Prompt ----------------------
def analyze_image_with_prompt(image: Image.Image, prompt: str) -> str:
    """
    Sends the prompt (plus optional base64-encoded image) to the custom model.
    """
    client = load_model()
    
    # Encode image to base64 if provided
    base64_image = encode_image_to_base64(image) if image else None

    # Prepare the message content
    content_list = [{"type": "text", "text": prompt}]
    if base64_image:
        content_list.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{base64_image}"}
        })

    # Call your custom model (named "gpt-4o-mini" in the reference)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": content_list  # List of content dicts
            }
        ],
        max_tokens=300
    )

    return response.choices[0].message.content

# ---------------------- Streamlit App ----------------------
def main():
    st.title("Simple Image+Text Chatbot with SQLite")
    st.write("A Streamlit chatbot that can optionally handle images and stores conversations in SQLite.")

    # Prompt input
    prompt = st.text_area("Enter your prompt:", key="user_prompt")

    # Image upload (optional)
    uploaded_file = st.file_uploader("Upload an image (optional)", type=["png", "jpg", "jpeg"], key="uploaded_image")
    image_obj = None
    if uploaded_file:
        try:
            image_obj = Image.open(uploaded_file)
        except Exception as e:
            st.error(f"Error reading image: {e}")
            image_obj = None

    # Submit button
    if st.button("Send"):
        if not prompt.strip() and not image_obj:
            st.warning("Please provide a prompt or an image.")
        else:
            with st.spinner("Analyzing..."):
                try:
                    bot_response = analyze_image_with_prompt(image_obj, prompt)
                except Exception as e:
                    bot_response = f"Error: {str(e)}"

            st.subheader("Response:")
            st.write(bot_response)

            # Save to database
            # If there's an image, encode it; otherwise set to None
            base64_img = encode_image_to_base64(image_obj) if image_obj else None
            save_conversation(prompt, base64_img, bot_response)

    # Sidebar: Load all past conversations
    st.sidebar.header("View Stored Conversations")
    if st.sidebar.button("Load All"):
        all_convos = load_conversations()
        if all_convos:
            for idx, convo in enumerate(all_convos, start=1):
                user_prompt, base64_img, bot_resp, timestamp = convo
                st.sidebar.markdown(f"**{idx}. Timestamp**: {timestamp}")
                st.sidebar.markdown(f"**User Prompt**: {user_prompt}")
                if base64_img:
                    # Convert base64 back to an image
                    try:
                        img_data = base64.b64decode(base64_img)
                        img = Image.open(io.BytesIO(img_data))
                        st.sidebar.image(img, caption="Uploaded Image", use_column_width=True)
                    except Exception:
                        pass
                st.sidebar.markdown(f"**Bot Response**: {bot_resp}")
                st.sidebar.write("---")
        else:
            st.sidebar.write("No conversations found.")

if __name__ == "__main__":
    main()
