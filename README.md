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

## Linear relationship and input ##

```
contrl = ( player_y_array - next_pipe_dist_to_player_array )
coff_0 = ( player_y_array - ( next_pipe_top_y_array + next_pipe_bottom_y_array - next_pipe_top_y_array ) )
coff_1 = ( player_y_array + distance_accum )
coff_2 = 300 + gamescores
coff_3 = 1
coff_4 = 1
coff_5 = 1
coff_6 = 1
coff_7 = 1
coff_8 = 1
coff_9 = 1
coff_10 = 1
coff_11 = 1
	
DATA_row = tf.constant([ contrl, coff_0, coff_1, coff_2, coff_3, coff_4, coff_5, coff_6, coff_7, coff_8, coff_9, coff_10, coff_11,
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ], shape=(1, 1, 1, 30), dtype=tf.float32)
```

## Training ##

![Alt text](https://github.com/jkaewprateep/Accelearate_linear_training/blob/main/ezgif.com-gif-maker%20(11).gif?raw=true "Title")


## Result ##

![Alt text](https://github.com/jkaewprateep/Accelearate_linear_training/blob/main/FlappyBird_small.gif?raw=true "Title")
