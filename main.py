import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pickle
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter
import numpy as np

# Download NLTK stopwords
nltk.download('stopwords', quiet=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model, tokenizer, and label encoder
@st.cache_resource
def load_models():
    model = AutoModelForSequenceClassification.from_pretrained("mental_health_model").to(device)
    tokenizer = AutoTokenizer.from_pretrained("mental_health_model")
    label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
    return model, tokenizer, label_encoder

model, tokenizer, label_encoder = load_models()

# Preprocessing
stop_words = set(stopwords.words("english"))

def clean_statement(statement):
    statement = statement.lower()
    statement = re.sub(r"[^\w\s]", "", statement)
    statement = re.sub(r"\d+", "", statement)
    words = statement.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

def split_text_into_chunks(text, max_length=200, overlap=50):
    """Split long text into overlapping chunks for better analysis"""
    words = text.split()
    chunks = []
    
    if len(words) <= max_length:
        return [text]
    
    for i in range(0, len(words), max_length - overlap):
        chunk = " ".join(words[i:i + max_length])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks

def analyze_text(text):
    """Analyze text and return detailed predictions"""
    cleaned_text = clean_statement(text)
    
    # Handle empty text
    if not cleaned_text.strip():
        return None, None, None
    
    # Split into chunks if text is long
    chunks = split_text_into_chunks(cleaned_text, max_length=180, overlap=40)
    
    all_predictions = []
    all_probabilities = []
    
    # Analyze each chunk
    for chunk in chunks:
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=200
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        predicted_class = torch.argmax(logits, dim=1).cpu().item()
        
        all_predictions.append(predicted_class)
        all_probabilities.append(probabilities)
    
    # Aggregate results
    avg_probabilities = np.mean(all_probabilities, axis=0)
    final_prediction = np.argmax(avg_probabilities)
    confidence = avg_probabilities[final_prediction]
    
    # Get prediction distribution
    prediction_counts = Counter(all_predictions)
    
    return final_prediction, confidence, prediction_counts

# Streamlit UI
st.set_page_config(page_title="Mental Health Analyzer", page_icon="🧠", layout="wide")

st.title("🧠 Advanced Mental Health Status Analyzer")
st.markdown("### Analyze text of any length to detect mental health status")

# Input section
st.subheader("Enter Your Text")
input_method = st.radio("Choose input method:", ["Text Input", "Text Area"], horizontal=True)

if input_method == "Text Input":
    input_text = st.text_input("Enter a short statement:", placeholder="I feel anxious about everything...")
else:
    input_text = st.text_area(
        "Enter your thoughts, notes, or journal entry:", 
        height=200,
        placeholder="Write as much as you'd like. This analyzer can handle both short statements and long journal entries..."
    )

# Analysis section
col1, col2 = st.columns([1, 4])
with col1:
    analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
with col2:
    if input_text:
        word_count = len(input_text.split())
        char_count = len(input_text)
        st.info(f"📊 Text stats: {word_count} words, {char_count} characters")

# Results section
if analyze_button:
    if not input_text.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing your text..."):
            predicted_class, confidence, prediction_dist = analyze_text(input_text)
            
            if predicted_class is None:
                st.error("Could not analyze the text. Please try again.")
            else:
                predicted_label = label_encoder.inverse_transform([predicted_class])[0]
                
                st.success("✅ Analysis Complete!")
                
                # Display results in columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Detected Status", predicted_label)
                
                with col2:
                    st.metric("Confidence", f"{confidence * 100:.1f}%")
                
                with col3:
                    chunks_analyzed = sum(prediction_dist.values())
                    st.metric("Text Segments Analyzed", chunks_analyzed)
                
                # Confidence interpretation
                st.markdown("---")
                st.subheader("Analysis Details")
                
                if confidence > 0.8:
                    confidence_text = "Very High - The model is quite certain about this prediction."
                elif confidence > 0.6:
                    confidence_text = "High - The model has good confidence in this prediction."
                elif confidence > 0.4:
                    confidence_text = "Moderate - There is some uncertainty in the prediction."
                else:
                    confidence_text = "Low - The prediction has significant uncertainty. Consider providing more context."
                
                st.write(f"**Confidence Level:** {confidence_text}")
                
                # Show distribution if multiple chunks were analyzed
                if len(prediction_dist) > 1:
                    st.markdown("**Segment-wise Analysis:**")
                    for pred_class, count in prediction_dist.most_common():
                        label = label_encoder.inverse_transform([pred_class])[0]
                        percentage = (count / sum(prediction_dist.values())) * 100
                        st.write(f"- {label}: {count} segments ({percentage:.1f}%)")
                
                # Disclaimer
                st.markdown("---")
                st.caption("⚠️ **Disclaimer:** This is an AI-based tool for informational purposes only. It is not a substitute for professional mental health advice, diagnosis, or treatment. If you're experiencing mental health concerns, please consult a qualified healthcare professional.")

# Sidebar information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This tool uses advanced natural language processing to analyze text for mental health indicators.
    
    **Features:**
    - Handles text of any length
    - Chunks long text for accurate analysis
    - Provides confidence scores
    - Segment-wise breakdown
    
    **Best Practices:**
    - Be honest and detailed
    - Write naturally
    - Include context when possible
    - Longer text = more accurate results
    """)
    
    st.markdown("---")
    st.markdown(f"**Model Device:** {device}")
    st.markdown(f"**Available Labels:** {len(label_encoder.classes_)}")