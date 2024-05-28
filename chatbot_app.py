import sys
import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import re
import unicodedata
from typing import List, Tuple, Dict
import customtkinter as ctk
from tkinter import StringVar, Text
import random


##################################
#### GRAPHICAL USER INTERFACE ####
##################################

# Initialize the customtkinter
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

class ChatbotGuiApp(ctk.CTk):
    """Class for Customtkinter based Graphical User Interface"""

    def __init__(self, model_instance):
        super().__init__()

        # Get Seq2Seq Model instance
        self.model = model_instance

        # Configure the main window
        self.title("Chatbot")
        self.geometry("1200x900")
        self.font_size = 18

        # Create a frame to hold the left and right parts
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True)

        # Left part of the main window
        self.left_frame = ctk.CTkFrame(self.main_frame)
        self.left_frame.grid(row=0, column=0, sticky="nsew")

        self.conversation_label = ctk.CTkLabel(self.left_frame, text="   Conversation", font=("Helvetica", 18, "bold"))
        self.conversation_label.pack(pady=10, anchor="w")

        self.conversation_text_frame = ctk.CTkFrame(self.left_frame)
        self.conversation_text_frame.pack(padx=10, pady=(0, 10), fill="both", expand=True)

        self.conversation_text = ctk.CTkTextbox(self.conversation_text_frame)
        self.conversation_text.pack(side="left", fill="both", expand=True)
        self.conversation_text.configure(font=("Arial", self.font_size))

        self.message_label = ctk.CTkLabel(self.left_frame, text="   Your message", font=("Helvetica", 18, "bold"))
        self.message_label.pack(pady=5, anchor="w")

        self.message_entry = ctk.CTkEntry(self.left_frame)
        self.message_entry.pack(padx=10, pady=5, fill="x", expand=False)
        self.message_entry.configure(font=("Arial", self.font_size))

        self.button_frame = ctk.CTkFrame(self.left_frame)
        self.button_frame.pack(pady=10, fill="x")

        self.send_button = ctk.CTkButton(self.button_frame, text="Send", command=self.send_message, font=("Helvetica", 18, "bold"))
        self.send_button.pack(side="left", padx=10, pady=10)

        self.random_button = ctk.CTkButton(self.button_frame, text="Random", command=self.random_message, font=("Helvetica", 18, "bold"))
        self.random_button.pack(side="right", padx=10, pady=10)

        # Bind the Enter key to the send_message function
        self.message_entry.bind('<Return>', self.on_enter_key_pressed)

        # Right part of the main window
        self.right_frame = ctk.CTkFrame(self.main_frame)
        self.right_frame.grid(row=0, column=1, sticky="nsew")

        self.random_samples_label = ctk.CTkLabel(self.right_frame, text="   Random samples from input dataset",
                                                 font=("Helvetica", 18, "bold"))
        self.random_samples_label.pack(pady=10, anchor="w")

        self.random_samples_text_frame = ctk.CTkFrame(self.right_frame)
        self.random_samples_text_frame.pack(padx=10, pady=10, fill="both", expand=True)

        self.random_samples_text = ctk.CTkTextbox(self.random_samples_text_frame, height=15)
        self.random_samples_text.configure(font=("Arial", self.font_size - 1, "normal"))
        self.random_samples_text.pack(side="left", fill="both", expand=True)

        self.good_examples_label = ctk.CTkLabel(self.right_frame, text="   Good performing examples",
                                                font=("Helvetica", 18, "bold"))
        self.good_examples_label.pack(pady=10, anchor="w")

        self.good_examples_text_frame = ctk.CTkFrame(self.right_frame)
        self.good_examples_text_frame.pack(padx=10, pady=10, fill="both", expand=True)

        self.good_examples_text = ctk.CTkTextbox(self.good_examples_text_frame, height=15)
        self.good_examples_text.configure(font=("Arial", self.font_size - 1, "normal"))
        self.good_examples_text.pack(side="left", fill="both", expand=True)

        # Configure grid weights to ensure proper resizing
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)

        # Insert start message
        self.append_to_conversation("Chatbot is ready! Type 'exit' to end the conversation.\n")

        # Fill in 'Random samples from input dataset' and 'Good performing examples' text boxes
        self.good_performing_examples = [
            'Good morning, Ladies!',
            'Did you drink a lot yesterday?',
            'What sports do you like?',
            'Do you believe in God?',
            'Is she very beautiful?',
            'Are you going by train?',
            'How long will your trip take?',
            'Why do you have a headache?',
            'May I help you?',
            'Do you have a lot of work?',
            'I love You!',
            'I hate you',
            'Can you help me with my homework?',
            'Where are you from?',
            'Tell me a story',
            "What's the meaning of life?",
            "What's your favorite hobby?",
            "You will recover after a good night 's sleep'"
        ]
        self.random_samples_dataset = self.load_dataframe_samples()
        self.append_to_random_samples(text for text in self.random_samples_dataset)
        self.append_to_good_examples(text for text in self.good_performing_examples)

        self.all_samples_for_random_button = self.good_performing_examples + self.random_samples_dataset


    def load_dataframe_samples(self) -> list:
        # Loading the DataFrame
        data_dir = os.path.join(os.getcwd(), 'data')
        file_path_parquet = os.path.join(data_dir, 'training_df_dd_tf210.parquet')
        training_data_final = pd.read_parquet(file_path_parquet)

        # Take 15 samples from the 'input' column
        samples = training_data_final['input'].sample(n=15).tolist()  # Convert to list
        return samples


    def format_response(self, response: str) -> str:
        response = response.strip()
        if response:
            if response[-1] not in ".!?":
                punctuation = random.choices([".", "!", " :)", "..."], [70, 10, 10, 10])[0]
                response += punctuation
        return response.capitalize()

    def send_message(self):
        # Functionality for send button
        user_input = self.message_entry.get()
        if user_input.strip():
            if user_input.lower() == 'exit':
                self.append_to_conversation("<Argama>:   Goodbye!")
                self.append_to_conversation("<Beamara>:   Goodbye!")
                self.after(3000, sys.exit)  # Wait 3 seconds before exiting
                return
            self.append_to_conversation(f"\n<You>:   {user_input}\n")
            self.message_entry.delete(0, 'end')

            input_text = self.model.preprocess_text(user_input)
            input_sequence = [self.model.tokenizer.texts_to_sequences([input_text])[0]]
            padded_input_sequence = pad_sequences(input_sequence, maxlen=self.model.max_length, padding='pre',
                                                  truncating='post')
            response = self.model.generate_response(np.array(padded_input_sequence), self.model.max_length)
            response_2 = self.model.beam_search_decode(np.array(padded_input_sequence), beam_width=3,
                                            max_decoder_seq_length=self.model.max_length)

            formatted_response = self.format_response(response)
            formatted_response_2 = self.format_response(response_2)

            self.append_to_conversation(f"<Argama>:   {formatted_response}")
            self.append_to_conversation(f"<Beamara>:   {formatted_response_2}")

    def on_enter_key_pressed(self, event):
        self.send_message()

    def random_message(self):
        # Functionality for random button
        random_message = random.choice(self.all_samples_for_random_button)
        self.message_entry.delete(0, 'end')
        self.message_entry.insert(0, random_message)

    def append_to_conversation(self, text: str):
        first_word = text.split()[0]
        if first_word == "<You>:":
            tag = "green"
            color = "green"
            alignment = "right"
        elif first_word == "<Argama>:":
            tag = "orange"
            color = "orange"
            alignment = "left"
        elif first_word == "<Beamara>:":
            tag = "pink"
            color = "pink"
            alignment = "left"
        else:
            tag = "default"
            color = "white"
            alignment = "left"

        self.conversation_text.tag_config(tag, foreground=color, justify=alignment)

        if alignment == "right":
            self.conversation_text.insert("end", text + "\n", (tag,))
            self.conversation_text.tag_add(tag, "end-1c linestart", "end-1c lineend")
            self.conversation_text.tag_config(tag, lmargin1=100, rmargin=10)  # Adjust margins for right alignment
        else:
            self.conversation_text.insert("end", text + "\n", (tag,))
            self.conversation_text.tag_add(tag, "end-1c linestart", "end-1c lineend")
            self.conversation_text.tag_config(tag, lmargin1=10, rmargin=100)  # Adjust margins for left alignment

        self.conversation_text.yview_moveto(1.0)

    def append_to_random_samples(self, samples):
        for text in samples:
            self.random_samples_text.insert("end", self.format_response(text) + "\n")
        self.random_samples_text.yview_moveto(1.0)

    def append_to_good_examples(self, samples):
        for text in samples:
            self.good_examples_text.insert("end", text + "\n")
        self.good_examples_text.yview_moveto(1.0)

##################################
####         THE MODEL        ####
##################################

class Seq2SeqModel:
    """Class for Sequence2Sequence Encoder-Decoder conversation generation model and its operations"""

    def __init__(self):
        # Configure Model and load weights

        # Print version information
        print(f"Python version: {sys.version}")
        print("TensorFlow version:", tf.__version__)
        print("TensorFlow Keras version:", tf.keras.__version__)

        # Get the current working directory
        self.cwd = os.getcwd()

        # Define paths
        self.tokenizer_path = os.path.join(self.cwd, 'data', 'tokenizer_dd_tf210.pickle')
        self.weights_path = os.path.join(self.cwd, 'data', 's2s_model_dd_tf210_weights.h5')

        # Load tokenizer from pickle
        with open(self.tokenizer_path, 'rb') as handle:
            self.tokenizer = pickle.load(handle)

        # Define model parameters
        self.latent_dim = 200
        self.num_tokens = len(self.tokenizer.word_index) + 1
        # self.num_tokens = self.tokenizer.num_words + 1  # Is used for 'data_alternative'

        self.learning_rate = 0.001

        # Define encoder
        self.encoder_inputs = Input(shape=(None,))
        self.encoder_embedding = Embedding(self.num_tokens, self.latent_dim, mask_zero=True)(self.encoder_inputs)
        self.encoder_lstm = Bidirectional(LSTM(self.latent_dim, return_state=True, dropout=0.2))
        self.encoder_outputs, self.forward_h, self.forward_c, self.backward_h, self.backward_c = self.encoder_lstm(self.encoder_embedding)
        self.state_h = Concatenate()([self.forward_h, self.backward_h])
        self.state_c = Concatenate()([self.forward_c, self.backward_c])
        self.encoder_states = [self.state_h, self.state_c]

        # Update latent_dim to match the concatenated states
        self.latent_dim *= 2

        # Define decoder
        self.decoder_inputs = Input(shape=(None,))
        self.decoder_embedding = Embedding(self.num_tokens, self.latent_dim, mask_zero=True)
        self.decoder_embedded = self.decoder_embedding(self.decoder_inputs)
        self.decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True, dropout=0.2)
        self.decoder_outputs, _, _ = self.decoder_lstm(self.decoder_embedded, initial_state=self.encoder_states)
        self.decoder_dense = Dense(self.num_tokens, activation='softmax')
        self.decoder_outputs = self.decoder_dense(self.decoder_outputs)

        # Define the model
        self.model = Model([self.encoder_inputs, self.decoder_inputs], self.decoder_outputs)

        # Compile the model
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='sparse_categorical_crossentropy')

        # Load model weights
        self.model.load_weights(self.weights_path)

        # Define encoder model for inference
        self.encoder_model = Model(self.encoder_inputs, self.encoder_states)

        # Define decoder model for inference
        self.decoder_state_input_h = Input(shape=(self.latent_dim,))
        self.decoder_state_input_c = Input(shape=(self.latent_dim,))
        self.decoder_states_inputs = [self.decoder_state_input_h, self.decoder_state_input_c]
        self.decoder_inference_inputs = Input(shape=(None,))
        self.decoder_embedding_inference = self.decoder_embedding(self.decoder_inference_inputs)
        self.decoder_outputs, self.state_h, self.state_c = self.decoder_lstm(
            self.decoder_embedding_inference, initial_state=self.decoder_states_inputs
        )
        self.decoder_states = [self.state_h, self.state_c]
        self.decoder_outputs = self.decoder_dense(self.decoder_outputs)
        self.decoder_model = Model(
            [self.decoder_inference_inputs] + self.decoder_states_inputs,
            [self.decoder_outputs] + self.decoder_states
        )

        # Save token index mappings
        self.target_token_index: Dict[str, int] = self.tokenizer.word_index
        self.reverse_target_token_index: Dict[int, str] = {v: k for k, v in self.target_token_index.items()}

        self.max_length = 15

        self.contractions = {
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


    def normalize_text(self, text: str) -> str:
        """Normalize Unicode string to NFKD form, remove non-ASCII characters, and lowercase."""
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
        text = text.lower()
        text = re.sub(r"\s*'\s*", "'", text)
        text = re.sub(r"\s*([.!?])\s*", r" \1 ", text)
        for contraction, replacement in self.contractions.items():
            text = re.sub(re.escape(contraction), replacement, text)
        text = re.sub(r"[^a-z' ]", ' ', text)
        text = re.sub(r"\s+", ' ', text).strip()
        return text


    def preprocess_text(self, text: str) -> str:
        """Preprocess the input text by normalizing and trimming to the maximum length."""
        text = self.normalize_text(text)
        words = text.split()[:self.max_length]
        return ' '.join(words)


    def generate_response(self, input_seq: np.ndarray, max_decoder_seq_length: int) -> str:
        """Generate a response using the trained model."""
        states_value = self.encoder_model.predict(input_seq, verbose=0)
        target_seq = np.ones((1, 1))
        target_seq[0, 0] = self.tokenizer.word_index['<START>']

        stop_condition = False
        self.decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value, verbose=0)
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.reverse_target_token_index[sampled_token_index]
            self.decoded_sentence += ' ' + sampled_char

            if (sampled_char == '<END>' or len(self.decoded_sentence.split()) > max_decoder_seq_length):
                stop_condition = True

            target_seq = np.ones((1, 1))
            target_seq[0, 0] = sampled_token_index
            states_value = [h, c]

        return self.decoded_sentence.strip().replace('<START>', '').replace('<END>', '').strip()


    def beam_search_decode(self, input_seq: np.ndarray, beam_width: int = 3, max_decoder_seq_length: int = 15) -> str:
        """Perform beam search decoding to generate a response."""
        states_value = self.encoder_model.predict(input_seq, verbose=0)
        start_token_index = self.tokenizer.word_index['<START>']
        end_token_index = self.tokenizer.word_index['<END>']
        beams = [(np.array([[start_token_index]]), states_value, 0.0)]

        for _ in range(max_decoder_seq_length):
            all_candidates = []
            for seq, states, score in beams:
                if seq[0, -1] == end_token_index:
                    all_candidates.append((seq, states, score))
                    continue

                output_tokens, h, c = self.decoder_model.predict([seq[:, -1:]] + states, verbose=0)
                top_k_indices = np.argsort(output_tokens[0, -1, :])[-beam_width:]

                for idx in top_k_indices:
                    new_seq = np.hstack([seq, np.array([[idx]])])
                    new_score = score + np.log(output_tokens[0, -1, idx])
                    all_candidates.append((new_seq, [h, c], new_score))

            beams = sorted(all_candidates, key=lambda x: x[2], reverse=True)[:beam_width]

            if all(seq[0, -1] == end_token_index for seq, _, _ in beams):
                break

        best_seq, _, _ = beams[0]
        self.decoded_sentence = ' '.join(
            [self.reverse_target_token_index[idx] for idx in best_seq[0] if idx != start_token_index and idx != end_token_index]
        )
        return self.decoded_sentence


if __name__ == "__main__":
    model = Seq2SeqModel()
    app = ChatbotGuiApp(model)
    app.mainloop()