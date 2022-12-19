# Accelearate_linear_training
Accelerates linear relationship equation

## Up-Samping and MaxPool technique ##

```
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Model Initialize
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
input_shape = (1, 30)

model = tf.keras.models.Sequential([
	tf.keras.layers.InputLayer(input_shape=input_shape),
	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True, return_state=False)),
	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
	tf.keras.layers.Conv1D(32, 1, activation='relu',input_shape=input_shape[1:]),
	tf.keras.layers.UpSampling1D(size=4),
	tf.keras.layers.Normalization(axis=-1),
	tf.keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='valid'),
	tf.keras.layers.UpSampling1D(size=2),
	tf.keras.layers.Normalization(axis=-1),
	tf.keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='valid'),
	tf.keras.layers.Dense(256),
	tf.keras.layers.Dropout(.2, input_shape=(256,))
])
```

## Training ##

![Alt text](https://github.com/jkaewprateep/Accelearate_linear_training/blob/main/ezgif.com-gif-maker%20(11).gif?raw=true "Title")


## Result ##

![Alt text](https://github.com/jkaewprateep/Accelearate_linear_training/blob/main/FlappyBird_small.gif?raw=true "Title")
