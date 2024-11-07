from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
import time

# ****** Callbacks ******
tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()),
                          histogram_freq=1,
                          write_graph=True,
                          write_images=True)

# Exercise 1
# Construire un MLP avec 2 couches cachées avec 200 neurones chacune et fonction d’activation de type ‘relu’ et une couche de sortie de type ‘softmax’.
# Ce réseau traite des entrées de dimension 400 et propose 10 valeurs en sortie.

nb_dimensions_input = 784

# Init sequential model
model = Sequential()

model.add(Dense(300, input_shape=(nb_dimensions_input,), activation='relu'))

# Adding second hidden layer
model.add(Dense(300, activation='relu'))

# Adding two more hidden layer (exercise 6)z
model.add(Dense(150, activation='relu'))
model.add(Dense(75, activation='relu'))

# Adding out layer
model.add(Dense(10, activation='softmax'))

# Display model structure
model.summary()

# Exercise 2
# Compiler le MLP en y associant une fonction de coût de type cross entropie catégorielle et un optimiseur de type descente stochastique du gradient (SDG).
# Il sera probablement nécessaire d’importer des packages de Keras.

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

# Exercise 3
# Que signifie ces deux lignes ?
# Trouvez la bonne valeur de nb_dimensions_entree.
# Ces lignes permettent de transformer les images du jeu de données MNIST en un format que le MLP peut traiter.
# Chaque image dans MNIST est de taille 28x28 pixels, donc la valeur de nb_dimensions_entree est 28 * 28 = 784.
# Vous pouvez afficher le contenu des différentes structures de données impliquées ici.

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("x_train dimension:", x_train.shape)      # (60000, 28, 28)
print("x_test dimensions:", x_test.shape)       # (10000, 28, 28)
print("y_train dimensions:", y_train.shape)     # (60000,)
print("y_test dimensions:", y_test.shape)       # (10000,)

x_train = x_train.reshape(60000, nb_dimensions_input)
x_test = x_test.reshape(10000, nb_dimensions_input)

print("x_train dimensions after reshaping:", x_train.shape)  # (60000, 784)
print("x_test dimensions after reshaping:", x_test.shape)    # (10000, 784)

# Exercise 4
# Quelle valeur doit être affectée à la variable nb_classes ?
# Le nombre de classes dans le jeu de données MNIST est 10, car il y a 10 chiffres (0 à 9).

nb_classes = 10
y_train = to_categorical(y_train, nb_classes)
y_test = to_categorical(y_test, nb_classes)

# Exercise 5
# Procéder à l’apprentissage et à l’évaluation de ce premier perceptron multi-couches (MLP).
# Modifier le code si nécessaire

# Launch the learning
model.fit(x_train, y_train, epochs=12, verbose=1, validation_split=0.1, callbacks=[tensorboard])

# Performances evaluation on data test
score = model.evaluate(x_test, y_test, verbose=0)
print('Test score:', score[0])      # Test score: 2.301025152206421
print('Test accuracy:', score[1])   # Test accuracy: 0.11349999904632568

# Exercise 6
# En agissant sur les différents paramètres à votre disposition, construire le système le plus performant possible.
# (optimiseur et paramètres liés à l’optimiseur, topologie de l’architecture neuronale, fonction de coût, fonction de régularisation, de normalisation...)

# Choosing adam as optimizer :
# Test score: 0.1538667529821396
# Test accuracy: 0.9656000137329102

# Adding tree more layers :
# Test score: 0.1394253969192505
# Test accuracy: 0.9729999899864197

# Adding more neurones to each layer :
# Test score: 0.12010600417852402
# Test accuracy: 0.9761000275611877
