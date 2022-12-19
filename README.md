# Accelearate_linear_training

Accelerates linear relationship equation training, these variables is slected by intention to study about linear relationship among 2 equations with non linear relation. Inputs are from the same equations, distance and player_y_array when player Y increase distance also increase but turnback point to continue player_Y_array decrease when second equation is stabelize with the first equation when first increase its second equation decrease. The third and fourth is running number to indicated longer objective.

### There are 2 objectives ####

1. Go though the gap between floor and ceiling to survive.
2. Player continue to continue adding into the running number.

## Up-Samping and MaxPool technique ## 

👧💬 It is you concatinate input of your series with same value then remove of less significant, it is linearly function. Leave those complexing method and think it this way first networks learn this way then you finish try more complex networks.

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

There are 2 objectives long and short term when short term passing the ceiling and floor gap and longer term to continue with running number.
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

### Files and directory ###

1. ezgif.com-gif-maker (11).gif : record player during the training process.
2. FlappyBird_small.gif : the result.
3. README.md : readme file.

## Training ##

You can see that start the scipt it start learning not joining the same point mistakes but when adjusting parameters to fit the equation may several times repeat since we keep try both conditions of games objective achivement and networks long run without overfittings as our AI working continue everyday.

![Alt text](https://github.com/jkaewprateep/Accelearate_linear_training/blob/main/ezgif.com-gif-maker%20(11).gif?raw=true "Title")

The finish it will acclerate fly without tired, this is AI.

## Result ##

![Alt text](https://github.com/jkaewprateep/Accelearate_linear_training/blob/main/FlappyBird_small.gif?raw=true "Title")
