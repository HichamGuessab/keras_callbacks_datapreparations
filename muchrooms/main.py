import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2

# loading data
df = pd.read_csv("champignons.csv")

x = df[["cap-diameter", "cap-shape", "gill-attachment", "gill-color", "stem-height", "stem-width", "stem-color", "season"]]
y = df["class"]

# first division : learning & temporary
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.2, random_state=42)

# second division : validation et test
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# data normalization
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

# building the neural network with regularisation & dropout
model = Sequential([
    Dense(128, activation='relu', input_shape=(x_train.shape[1],), kernel_regularizer=l2(0.01)),
    Dropout(0.3),  # mask 30% of neurons to avoid over learning
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(1, activation='sigmoid') # for a binary classification
])

# compiling the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# training the model
model.fit(x_train, y_train, epochs=120, batch_size=64, validation_data=(x_val, y_val))

# evaluating the model
test_score = model.evaluate(x_test, y_test)
print("Test accuracy:", test_score[1])
# 30 epochs  -> Test accuracy: 0.8817542791366577 -> better than previous
# 60 epochs  -> Test accuracy: 0.9241302609443665
# 120 epochs -> Test accuracy: 0.943375289440155