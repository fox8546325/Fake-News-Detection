
import joblib

# Load trained model
model = joblib.load("models/fake_news_model.pkl")

print("================================")
print("      Fake News Detection")
print("================================")

# User input
title = input("\nEnter news title: ")

text = input("\nEnter news content: ")

# Combine title and content
news = title + " " + text

# Make prediction
prediction = model.predict([news])[0]

# Get prediction probability
probability = model.predict_proba([news])[0]

print("\n========== Prediction ==========")

if prediction == 0:
    print("Result: FAKE NEWS")
else:
    print("Result: REAL NEWS")

print(f"Fake probability: {probability[0]:.2%}")
print(f"Real probability: {probability[1]:.2%}")

print("================================")
