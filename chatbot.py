import tkinter as tk
from tkinter import scrolledtext
import json
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load FAQ Data

with open("faq_data.json", "r") as file:
    faq_data = json.load(file)

questions = [item["question"] for item in faq_data]
answers = [item["answer"] for item in faq_data]

# Text Preprocessing

stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    return " ".join(tokens)

processed_questions = [preprocess(q) for q in questions]

# TF-IDF Vectorizer

vectorizer = TfidfVectorizer()

question_vectors = vectorizer.fit_transform(processed_questions)

# Chatbot Response Function

def get_response(user_input):
    processed_input = preprocess(user_input)

    input_vector = vectorizer.transform([processed_input])

    similarity = cosine_similarity(input_vector, question_vectors)

    best_match = np.argmax(similarity)

    score = similarity[0][best_match]

    if score > 0.3:
        return answers[best_match]
    else:
        return "Sorry, I couldn't understand your question. Please try asking differently."

# Send Message

def send_message():
    user_message = entry_box.get()

    if user_message.strip() == "":
        return

    chat_area.config(state=tk.NORMAL)

    # Display user message
    chat_area.insert(tk.END, "You: " + user_message + "\n\n")

    # Get bot response
    response = get_response(user_message)

    # Display bot response
    chat_area.insert(tk.END, "Surya Datta: " + response + "\n\n")

    chat_area.config(state=tk.DISABLED)

    # Auto-scroll
    chat_area.yview(tk.END)

    # Clear input
    entry_box.delete(0, tk.END)

# GUI Window

root = tk.Tk()
root.title("Surya Datta FAQ Chatbot")
root.geometry("700x650")
root.config(bg="#0f172a")

# Header

header = tk.Label(
    root,
    text="🤖 Surya Datta's FAQ Chatbot",
    font=("Helvetica", 22, "bold"),
    bg="#1e293b",
    fg="white",
    pady=15
)
header.pack(fill=tk.X)

# Chat Area

chat_area = scrolledtext.ScrolledText(
    root,
    wrap=tk.WORD,
    font=("Arial", 12),
    bg="#e2e8f0",
    fg="#0f172a",
    padx=15,
    pady=15
)

chat_area.pack(padx=15, pady=15, fill=tk.BOTH, expand=True)

chat_area.insert(
    tk.END,
    "Surya Datta: Hello 👋\nAsk me any FAQ related question!\n\n"
)

chat_area.config(state=tk.DISABLED)

# Bottom Frame

bottom_frame = tk.Frame(root, bg="#0f172a")
bottom_frame.pack(fill=tk.X, padx=10, pady=10)

# Entry Box
entry_box = tk.Entry(
    bottom_frame,
    font=("Arial", 14),
    bg="white",
    fg="black",
    relief=tk.FLAT
)

entry_box.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10), ipady=10)

# Send Button
send_button = tk.Button(
    bottom_frame,
    text="Send",
    font=("Arial", 12, "bold"),
    bg="#2563eb",
    fg="white",
    activebackground="#1d4ed8",
    activeforeground="white",
    padx=20,
    pady=10,
    relief=tk.FLAT,
    command=send_message
)

send_button.pack(side=tk.RIGHT)

# Enter Key Support
root.bind('<Return>', lambda event: send_message())
root.mainloop()