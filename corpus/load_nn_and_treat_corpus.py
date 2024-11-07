from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

nb_dimensions_input = 784

(x_train, y_train), (x_test, y_test) = mnist.load_data()
nb_classes = 10
y_test = to_categorical(y_test, nb_classes)

x_test = x_test.reshape(10000, nb_dimensions_input)

best_model = load_model("best_model.keras")
score = best_model.evaluate(x_test, y_test)

print("Test accuracy:", score[1])