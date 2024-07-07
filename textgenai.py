import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop
import os
import random

# Load your CSV data
current_d = os.getcwd()
print("Current Directory:", current_d)

# Assuming FakeRED.csv contains a column named 'text'
text_df = pd.read_csv("C:\\Users\\Administrator\\Desktop\\pythonprojects\\FakeRED.csv")
print(text_df.head())

# Concatenate all text into a single string for processing
joined_text = "".join(text_df['text'].values)
particle_text = joined_text[:100000]

# Tokenize the text
tokenizer = RegexpTokenizer(r"\w+")
tokens = tokenizer.tokenize(particle_text.lower())
print("Tokens:", tokens[:20])  # Print first 20 tokens as a sample

# Create a set of unique tokens and their index
unique_tokens = np.unique(tokens)
unique_tokens_index = {token: idx for idx, token in enumerate(unique_tokens)}
print("Number of unique tokens:", len(unique_tokens))

# Prepare input sequences and next words
n_words = 10
input_words = []
next_words = []

for i in range(len(tokens) - n_words):
    input_words.append(tokens[i:i + n_words])
    next_words.append(tokens[i + n_words])

print("Example input sequence:", input_words[0])
print("Example next word:", next_words[0])

# Convert input sequences and next words into one-hot encoding
x = np.zeros((len(input_words), n_words, len(unique_tokens)), dtype=bool)
y = np.zeros((len(next_words), len(unique_tokens)))

for i, words in enumerate(input_words):
    for j, word in enumerate(words):
        if word in unique_tokens_index:
            x[i, j, unique_tokens_index[word]] = 1
        else:
            print(f"Word '{word}' not found in unique_tokens_index. Skipping...")

    if next_words[i] in unique_tokens_index:
        y[i, unique_tokens_index[next_words[i]]] = 1
    else:
        print(f"Next word '{next_words[i]}' not found in unique_tokens_index. Skipping...")

# Build LSTM model
model = Sequential()
model.add(LSTM(128, input_shape=(n_words, len(unique_tokens))))
model.add(Dense(len(unique_tokens)))
model.add(Activation('softmax'))

# Compile model
model.compile(loss="categorical_crossentropy", optimizer=RMSprop(learning_rate=0.01), metrics=["accuracy"])

# Train model
model.fit(x, y, batch_size=128, epochs=30, shuffle=True)

# Save model in the current directory
model.save("mymodel.h5")
print("Model saved as mymodel.h5 in the current directory.")

# Load model from the current directory
loaded_model = load_model("mymodel.h5")
print("Model loaded successfully.")

# Function to predict next word
def predict_next_word(input_text, n_best):
    input_text = input_text.lower()
    x = np.zeros((1, n_words, len(unique_tokens)))
    for i, word in enumerate(input_text.split()):
        if word in unique_tokens_index:
            x[0, i, unique_tokens_index[word]] = 1
        else:
            print(f"Word '{word}' not found in unique_tokens_index. Using default value.")

    predictions = loaded_model.predict(x)[0]
    top_n_indices = np.argsort(predictions)[-n_best:][::-1]  # Get indices of top n predictions
    return top_n_indices

# Example prediction
possible = predict_next_word("The president of United states has announced that he", 5)
print("Possible next word indices:", possible)
print("Possible next words:", [unique_tokens[idx] for idx in possible])

# Function to generate text
def generate_text(input_text, text_length, creativity=3):
    word_sequence = input_text.split()
    current = 0
    for i in range(text_length):
        sub_sequence = " ".join(word_sequence[current:current + n_words])
        try:
            top_indices = predict_next_word(sub_sequence, creativity)
            choice = unique_tokens[random.choice(top_indices)]
        except Exception as e:
            print(f"Error predicting next word: {e}")
            choice = random.choice(unique_tokens)  # Handle unknown words by choosing randomly
        word_sequence.append(choice)
        current += 1
    return " ".join(word_sequence)

# Generate text example
generated_text = generate_text("He will have to look into this thing and he", 100, 5)
print("Generated Text:", generated_text)

# Continue with further model evaluation or deployment
# Note: The previous attempt to load from a different directory is removed as it's not relevant anymore.
