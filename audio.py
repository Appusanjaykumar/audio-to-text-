import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st
import re
import spacy
import speech_recognition as sr
from textblob import TextBlob
from googletrans import Translator
from io import BytesIO
import pydub
import io
import tempfile
import os
# Additional imports for evaluation metrics
from sklearn.metrics import accuracy_score
import spacy

# Download spaCy model 'en_core_web_sm'
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


# Function to calculate Word Error Rate (WER)
def calculate_wer(reference, hypothesis):
    reference_words = reference.split()
    hypothesis_words = hypothesis.split()

    if len(reference_words) == 0:
        # Handle the case where the reference transcription has no words
        return float('inf')

    insertions = len(set(hypothesis_words) - set(reference_words))
    deletions = len(set(reference_words) - set(hypothesis_words))
    substitutions = sum(1 for ref, hyp in zip(reference_words, hypothesis_words) if ref != hyp)

    wer = (insertions + deletions + substitutions) / len(reference_words)
    return wer


# Function to calculate Character Error Rate (CER)
def calculate_cer(reference, hypothesis):
    if len(reference) == 0:
        # Handle the case where the reference transcription has no characters
        return float('inf')

    insertions = len(set(hypothesis) - set(reference))
    deletions = len(set(reference) - set(hypothesis))
    substitutions = sum(1 for r, h in zip(reference, hypothesis) if r != h)

    cer = (insertions + deletions + substitutions) / len(reference)
    return cer


# Function to calculate accuracy
def calculate_accuracy(reference, hypothesis):
    if len(reference) == 0:
        # Handle the case where the reference transcription has no characters
        return float('inf')

    correct = sum(1 for r, h in zip(reference, hypothesis) if r == h)
    accuracy = correct / len(reference)
    return accuracy

def clean_and_tokenize(text):
    # Add your implementation here
    # For example, using regular expressions to tokenize
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens

# Function to transcribe and process audio
# Function to transcribe and process audio
def transcribe_and_process_audio(audio_file, reference_transcription):
    recognizer = sr.Recognizer()

    # Initialize variables
    transcription_result, processed_tokens, sentiment_result, ner_result, translated_text = None, None, None, None, None

    # Create a temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
        temp_wav_path = temp_wav.name

        if not audio_file.name.endswith(('.wav')):  # Adjust the allowed audio file extensions
            st.write("Invalid file format. Please upload a supported audio file (e.g., WAV, MP3, OGG, FLAC).")
            return None, None, None, None, None

        try:
            audio_content = audio_file.read()

            # Save the audio data to the temporary WAV file
            with open(temp_wav_path, 'wb') as temp_wav_file:
                temp_wav_file.write(audio_content)

            audio_data = pydub.AudioSegment.from_wav(temp_wav_path)

            with sr.AudioFile(temp_wav_path) as source:
                audio_data = recognizer.record(source)

            transcription_result = recognizer.recognize_google(audio_data, language="mr-IN")
            st.write("Transcription:", transcription_result)

            # Rest of the code...
            processed_tokens = clean_and_tokenize(transcription_result)
            sentiment_result = perform_sentiment_analysis(transcription_result)
            ner_result = perform_ner(transcription_result)

            # Perform translation only once
            translated_text = perform_translation(transcription_result, target_language='en')

            # Display translated text
            st.write("Translated Text:", translated_text)

        except sr.UnknownValueError:
            st.write("Could not understand audio")
        except sr.RequestError as e:
            st.write(f"Error with the API request; {e}")
        finally:
            # Clean up temporary WAV file
            if os.path.exists(temp_wav_path):
                # Close the file to release any locks
                temp_wav.close()

                # Remove the file
                os.remove(temp_wav_path)

    return transcription_result, processed_tokens, sentiment_result, ner_result, translated_text



# Placeholder NLP Model
class MarathiTranscriptionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MarathiTranscriptionModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Take the last time step output
        return out

# Placeholder NLP Model
class YourNLPModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(YourNLPModel, self).__init__()
        # Define your NLP model architecture here (replace this with your actual model)
        self.embedding = nn.Embedding(input_size, 128)
        self.lstm = nn.LSTM(128, 64, batch_first=True)
        self.fc = nn.Linear(64, output_size)

    def forward(self, x):
        # Implement the forward pass of your NLP model
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out[:, -1, :])  # Use the last LSTM output
        return output

# Function for model training (replace with your NLP model training logic)
def train_model(your_model, train_loader, epochs, lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    your_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(your_model.parameters(), lr=lr)

    for epoch in range(epochs):
        your_model.train()
        total_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = your_model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        st.write(f"NLP Model Epoch {epoch + 1}/{epochs}, Loss: {average_loss}")

# Function for model evaluation (replace with your NLP model evaluation logic)
def evaluate_model(your_model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    your_model.to(device)
    your_model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = your_model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    return all_preds, all_labels

# Function for replicating results
def replicate_results():
    # Replace with your data loading and preprocessing logic
    # For example, you can use the preprocess_data function here
    # preprocessed_data = preprocess_data("your_data_path.csv", "text_column", "label_column")

    your_model = YourNLPModel(input_size, output_size)  # Define input_size and output_size

    train_dataset = TensorDataset(
        torch.tensor(preprocess_data['train_vectors'].toarray(), dtype=torch.float32),
        torch.tensor(preprocess_data['train_labels'], dtype=torch.long)
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    train_model(your_model, train_loader, epochs=5, lr=0.001)

    test_dataset = TensorDataset(
        torch.tensor(preprocess_data['test_vectors'].toarray(), dtype=torch.float32),
        torch.tensor(preprocess_data['test_labels'], dtype=torch.long)
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    predictions, true_labels = evaluate_model(your_model, test_loader)

    accuracy = accuracy_score(true_labels, predictions)
    classification_rep = classification_report(true_labels, predictions)

    st.write(f"Replicated Accuracy: {accuracy}")
    st.write("Replicated Classification Report:\n", classification_rep)

# Example definitions (replace with your actual implementations)
input_size = 100
output_size = 10

# Example data preprocessing
def preprocess_data(data_path, text_column, label_column):
    # Your implementation here (replace with actual data preprocessing logic)
    df = pd.read_csv(data_path)
    texts = df[text_column].tolist()
    labels = df[label_column].tolist()

    # Placeholder for TfidfVectorizer (replace with your actual vectorizer)
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectors = tfidf_vectorizer.fit_transform(texts)

    # Placeholder for label encoding (replace with your actual encoding logic)
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Split data into train and test sets
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        tfidf_vectors, encoded_labels, test_size=0.2, random_state=42
    )

    return {
        'tfidf_vectorizer': tfidf_vectorizer,
        'train_vectors': train_texts,
        'test_vectors': test_texts,
        'train_labels': train_labels,
        'test_labels': test_labels,
    }

# Placeholder for your actual functions
# Placeholder for sentiment analysis implementation
def perform_sentiment_analysis(text):
    # Your sentiment analysis implementation using TextBlob
    analysis = TextBlob(text)
    sentiment = analysis.sentiment.polarity

    # You can define your own criteria for sentiment classification based on the polarity
    if sentiment > 0:
        return "Positive"
    elif sentiment < 0:
        return "Negative"
    else:
        return "Neutral"

# Placeholder for named entity recognition (NER) implementation
def perform_ner(text):
    # Your named entity recognition implementation using spaCy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    # Extract named entities
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    return entities

# Placeholder for translation implementation
def perform_translation(text, target_language='en'):
    # Your translation implementation using Googletrans
    translator = Translator()
    translation_result = translator.translate(text, dest=target_language).text

    return translation_result

# Streamlit app
def main():
    st.title("Marathi Transcription Streamlit App")

    # File upload and user input
    uploaded_files = st.file_uploader("Upload your Marathi audio files", type=["wav"], accept_multiple_files=True)
    reference_transcription = st.text_input("Enter the reference transcription in Marathi")

    selected_file = st.multiselect("Select the audio file for transcription", [file.name for file in uploaded_files])

    if selected_file:
        selected_audio_file = [file for file in uploaded_files if file.name == selected_file[0]][0]
        st.write("### Transcription Results")
        transcription_result, processed_tokens, sentiment_result, ner_result, translated_text = transcribe_and_process_audio(selected_audio_file, reference_transcription)

        # Display results
        st.write("Transcription:", transcription_result)
        st.write("Processed Tokens:", processed_tokens)
        st.write("Sentiment:", sentiment_result)
        st.write("Named Entities:", ner_result)
        st.write("Translated Text:", translated_text)

        # Evaluate the transcription
        wer = calculate_wer(reference_transcription, transcription_result)
        cer = calculate_cer(reference_transcription, transcription_result)
        accuracy = calculate_accuracy(reference_transcription, transcription_result)

        st.write("\n### Evaluation Metrics:")
        st.write("Word Error Rate:", wer)
        st.write("Character Error Rate:", cer)
        st.write("Accuracy:", accuracy)

# Run the Streamlit app
if __name__ == "__main__":
    main()
