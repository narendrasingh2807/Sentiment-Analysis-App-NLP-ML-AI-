import streamlit as st 
import joblib 

# Load the pre_trained model
pipe_lr = joblib.load("emotion_classifier.pkl")

#Define function to predict emotions 
def predict_emotion(text):
    prediction = pipe_lr.predict([text])[0]
    return prediction

# Main function
def main():
    st.title("Emotion Classifier App")
    
    #Input text area
    text = st.text_area("Enter text")
    
    # Button to trigger prediction 
    if st.button("Predict"):
        # Predict emotion 
        emotion = predict_emotion(text)
        st.write(f"Predicted emotion: {emotion}")
        
# Run the app 
if __name__ == "__main__":
    main()