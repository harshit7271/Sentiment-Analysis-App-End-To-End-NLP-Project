import streamlit as st
import pickle
from PIL import Image

@st.cache_resource
def load_model_and_vectorizer():
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('SentimentAnalysis.pkl', 'rb') as f:
        model = pickle.load(f)
    return vectorizer, model

vectorizer, model = load_model_and_vectorizer()

st.set_page_config(page_title="Sentiment Analysis App", page_icon="😊", layout='centered')
st.title("💬 Sentiment Analysis Web Application")

st.markdown(
    """
    <div style="background-color:#f9f9f9;padding:10px;border-radius:10px;">
    <h3 style="color:#4a90e2;">Welcome!</h3>
    <p>Enter a piece of text below and our smart ML model will analyze its sentiment 🎉</p>
    </div>
    """, unsafe_allow_html=True
)

# User input box
user_input = st.text_area("Enter your feelings here ...", height=150)

if user_input:
    try:
        # Vectorize input and predict sentiment
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]

        # Show prediction 
        sentiment_icon = "✅" if prediction in [0, 'positive', 'happy'] else "⚠️"
        sentiment_color = "#a5c072" if prediction in [0, 'positive', 'happy'] else "#3cc5e7"

        st.markdown(f"""
            <div style="background-color:{sentiment_color};padding:20px;border-radius:15px;color:white;text-align:center;font-size:24px;">
            {sentiment_icon} Predicted Sentiment: <strong>{prediction}</strong>
            </div>
        """, unsafe_allow_html=True)

        # Show confidence 
        if hasattr(model, 'predict_proba'):
            confidences = model.predict_proba(input_vector)[0]
            max_confidence = max(confidences)
            st.write(f"💡 Confidence Score: {max_confidence:.2f}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Sidebar information and rating
st.sidebar.header("About")
st.sidebar.info(
    "This sentiment analysis app is trained on a dataset called as train.txt "
    "of approximately 16,000 texts. "
    "Built with Streamlit & scikit-learn 💻"
)
name = st.sidebar.text_input("Please enter your name")

with st.sidebar.expander("Rate Our Application"):
    rating = st.slider("Please rate our application on the basis of performance", 1, 5, 3)
    st.write(f"HEY {name}, You rated our application {rating} out of 5")



