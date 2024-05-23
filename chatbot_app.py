import sys
import os
import logging
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import re
import unicodedata
from typing import List, Tuple, Dict

# Print version information
print(f"Python version: {sys.version}")
print("TensorFlow version:", tf.__version__)
print("TensorFlow Keras version:", tf.keras.__version__)

# Get the current working directory
cwd = os.getcwd()

# Define paths
tokenizer_path = os.path.join(cwd, 'data', 'tokenizer_dd_tf210.pickle')
weights_path = os.path.join(cwd, 'data', 's2s_model_dd_tf210_weights.h5')

# Load tokenizer from pickle
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Define model parameters
latent_dim = 200
num_tokens = len(tokenizer.word_index) + 1
learning_rate = 0.001

# Define encoder
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(num_tokens, latent_dim, mask_zero=True)(encoder_inputs)
encoder_lstm = Bidirectional(LSTM(latent_dim, return_state=True, dropout=0.2))
encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm(encoder_embedding)
state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])
encoder_states = [state_h, state_c]

# Update latent_dim to match the concatenated states
latent_dim *= 2

# Define decoder
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(num_tokens, latent_dim, mask_zero=True)
decoder_embedded = decoder_embedding(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.2)
decoder_outputs, _, _ = decoder_lstm(decoder_embedded, initial_state=encoder_states)
decoder_dense = Dense(num_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile the model
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy')

# Load model weights
model.load_weights(weights_path)

# Define encoder model for inference
encoder_model = Model(encoder_inputs, encoder_states)

# Define decoder model for inference
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_inference_inputs = Input(shape=(None,))
decoder_embedding_inference = decoder_embedding(decoder_inference_inputs)
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_embedding_inference, initial_state=decoder_states_inputs
)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inference_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states
)

# Save token index mappings
target_token_index: Dict[str, int] = tokenizer.word_index
reverse_target_token_index: Dict[int, str] = {v: k for k, v in target_token_index.items()}

max_length = 15

contractions = {
    "’": "'",
    "‘": "'",
    "“": '"',
    "”": '"',
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "i'm": "i am",
    "i'd": "i would",
    "thats's": "that is",
    "it's": "it is",
    "he's": "he is",
    "she's": "she is",
    "you're": "you are",
    "they're": "they are",
    "we're": "we are",
    "i've": "i have",
    "you've": "you have",
    "they've": "they have",
    "we've": "we have",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "doesn't": "does not",
    "don't": "do not",
    "didn't": "did not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "i'll": "i will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "we'll": "we will",
    "they'll": "they will",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "couldn't": "could not",
    "mightn't": "might not",
    "mustn't": "must not",
    "she'd": "she would",
    "he'd": "he would",
    "they'd": "they would",
    "we'd": "we would",
    "that'll": "that will",
    "there'll": "there will",
    "who'll": "who will",
    "it'll": "it will",
    "that'd": "that would",
    "there'd": "there would",
    "who'd": "who would",
    "when's": "when is",
    "where's": "where is",
    "why's": "why is",
    "how's": "how is",
    "y'all": "you all",
    "let's": "let us",
    "ma'am": "madam",
    "o'clock": "of the clock",
    "ain't": "is not",
    "could've": "could have",
    "should've": "should have",
    "would've": "would have",
    "might've": "might have",
    "must've": "must have",
    "who've": "who have",
    "oughtn't": "ought not",
    "daren't": "dare not",
    "needn't": "need not",
    "what's": "what is",
    "usedn't": "used not"
}


def normalize_text(text: str) -> str:
    """Normalize Unicode string to NFKD form, remove non-ASCII characters, and lowercase."""
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = text.lower()
    text = re.sub(r"\s*'\s*", "'", text)
    text = re.sub(r"\s*([.!?])\s*", r" \1 ", text)
    for contraction, replacement in contractions.items():
        text = re.sub(re.escape(contraction), replacement, text)
    text = re.sub(r"[^a-z' ]", ' ', text)
    text = re.sub(r"\s+", ' ', text).strip()
    return text


def preprocess_text(text: str) -> str:
    """Preprocess the input text by normalizing and trimming to the maximum length."""
    text = normalize_text(text)
    words = text.split()[:max_length]
    return ' '.join(words)


def generate_response(input_seq: np.ndarray, max_decoder_seq_length: int) -> str:
    """Generate a response using the trained model."""
    states_value = encoder_model.predict(input_seq, verbose=0)
    target_seq = np.ones((1, 1))
    target_seq[0, 0] = tokenizer.word_index['<START>']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_token_index[sampled_token_index]
        decoded_sentence += ' ' + sampled_char

        if (sampled_char == '<END>' or len(decoded_sentence.split()) > max_decoder_seq_length):
            stop_condition = True

        target_seq = np.ones((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]

    return decoded_sentence.strip().replace('<START>', '').replace('<END>', '').strip()


def beam_search_decode(input_seq: np.ndarray, beam_width: int = 3, max_decoder_seq_length: int = 15) -> str:
    """Perform beam search decoding to generate a response."""
    states_value = encoder_model.predict(input_seq, verbose=0)
    start_token_index = tokenizer.word_index['<START>']
    end_token_index = tokenizer.word_index['<END>']
    beams = [(np.array([[start_token_index]]), states_value, 0.0)]

    for _ in range(max_decoder_seq_length):
        all_candidates = []
        for seq, states, score in beams:
            if seq[0, -1] == end_token_index:
                all_candidates.append((seq, states, score))
                continue

            output_tokens, h, c = decoder_model.predict([seq[:, -1:]] + states, verbose=0)
            top_k_indices = np.argsort(output_tokens[0, -1, :])[-beam_width:]

            for idx in top_k_indices:
                new_seq = np.hstack([seq, np.array([[idx]])])
                new_score = score + np.log(output_tokens[0, -1, idx])
                all_candidates.append((new_seq, [h, c], new_score))

        beams = sorted(all_candidates, key=lambda x: x[2], reverse=True)[:beam_width]

        if all(seq[0, -1] == end_token_index for seq, _, _ in beams):
            break

    best_seq, _, _ = beams[0]
    decoded_sentence = ' '.join(
        [reverse_target_token_index[idx] for idx in best_seq[0] if idx != start_token_index and idx != end_token_index]
    )
    return decoded_sentence


def chat() -> None:
    """Interactive chat function to engage with the chatbot."""
    print("Chatbot is ready! Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Argama: Goodbye!")
            print("Beamara: Goodbye!")
            break
        input_text = preprocess_text(user_input)
        input_sequence = [tokenizer.texts_to_sequences([input_text])[0]]
        padded_input_sequence = pad_sequences(input_sequence, maxlen=max_length, padding='pre', truncating='post')
        response = generate_response(np.array(padded_input_sequence), max_length)
        response_2 = beam_search_decode(np.array(padded_input_sequence), beam_width=3,
                                        max_decoder_seq_length=max_length)
        print(f"Argama: {response}")
        print(f"Beamara: {response_2}")


if __name__ == "__main__":
    chat()
