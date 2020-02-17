import tensorflow as tf
from tensorflow import keras
import numpy as np
import os.path
import re


def paddingData(data):
	paddingData = keras.preprocessing.sequence.pad_sequences(
		data, 
		value=word_index["<PAD>"], 
		padding="post", 
		truncating="post", 
		maxlen=256
	)
	return paddingData

# Covert digital array to readable text
def decode_review(encodedText):
	return " ".join([reverse_word_index.get(i, "?") for i in encodedText])

# Convert array text to digital array
def encode_review(array):
	encoded_array = [word_index["<START>"]]
	for letter in array:
		encoded_array.append(word_index.get(letter, word_index["<UNK>"]))
	return encoded_array


# Preprocess text data
vocabularySize = 88000
data = keras.datasets.imdb
word_index = data.get_word_index()
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=vocabularySize)

word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = {v: k for k, v in word_index.items()}

train_data = paddingData(train_data)
test_data = paddingData(test_data)

x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

model_name = 'Practice-2.h5'
if os.path.isfile(model_name):
	# Load model if model created
	model = keras.models.load_model(model_name)
else:
	# Create model
	model = keras.Sequential([
		keras.layers.Embedding(vocabularySize, 16),
		keras.layers.GlobalAveragePooling1D(),
		keras.layers.Dense(16, activation="relu"),
		keras.layers.Dense(1, activation="sigmoid")
	])

	model.summary()
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_val, y_val))

	model.evaluate(test_data, test_labels)

	# Save model
	model.save(model_name)

# Predict test data
# predict = model.predict(test_data)
# print("Review: ")
# print(decode_review(test_data[8]))
# print("Predict: " + str(predict[8]))
# print("Actual: " + str(test_labels[8]))

# Predict the real data
review = ''
with open("Practice-2.txt") as f:
	for line in f.readlines():
		review += line.strip() + ' '

review = re.sub(r'[^ a-zA-Z0-9_]', '', review)
review = review.lower().strip().split(" ")
print(review)
encode = encode_review(review)
print(encode)
encode = paddingData([encode])
predict = model.predict(encode)
print(predict)