import tensorflow as tf
import numpy as np

celsius_q    = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit_q = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

layer0 = tf.keras.layers.Dense(units=1, input_shape = [1])
model = tf.keras.Sequential([layer0])
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
history = model.fit(celsius_q, fahrenheit_q, epochs=500, verbose=False)

celsius = int(input("Enter celsius: "))
print(f"{celsius} celsius = {model.predict([celsius, 29])} fahrenheit")