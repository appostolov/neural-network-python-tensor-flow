import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import config
import numpy as np
from src.neural_network.vocab.vocab import Vocab

class Network:
    def __init__(self, vocab_size: int, max_length: int = config.EMAIL_MAX_LEN):

        self.max_length = max_length
        self.vocab_size = vocab_size

        self.model = models.Sequential([
            layers.Embedding(
                input_dim=self.vocab_size, 
                output_dim=config.NN_EMBEDDING_OUT, 
                input_length=self.max_length
            ),
            layers.Bidirectional(
                layers.LSTM(config.NN_LSTM_HID, return_sequences=False)
            ),
            layers.Dense(config.NN_HID_SIZE, activation='relu'),
            layers.Dropout(config.NN_DROP),
            layers.Dense(1, activation='sigmoid')
        ])

        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    def dataset(file: str = config.DATASET_PATH):

        df = pd.read_csv(file)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        vocab = Vocab()

        inputs = np.array([vocab.text_to_indexes(email, config.EMAIL_MAX_LEN) for email in df['email'].values])
        labels = df['label'].values.astype(np.float32)

        return inputs, labels
