import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# loading data
df = pd.read_csv("champignons.csv")

x = df[["cap-diameter", "cap-shape", "gill-attachment", "gill-color", "stem-height", "stem-width", "stem-color", "season"]]
y = df["class"]

# first division : learning & temporary
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.2, random_state=42)

# second division : validation et test
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# building the neural network
model = Sequential([
    Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid') # for a binary classification
])

# compiling the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# training the model
model.fit(x_train, y_train, epochs=20, validation_data=(x_val, y_val))

# evaluating the model
test_score = model.evaluate(x_test, y_test)
print("Test accuracy:", test_score[1]) # Test accuracy: 0.730384886264801