import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, Concatenate, Flatten
from keras.models import Model

def build_model(grid_world_shape):
  inputs = Input(shape=grid_world_shape)

  inputs = Flatten()(inputs)
  x = Dense(32, activation='relu')(inputs)
  x = Dense(16, activation='relu')(x)
  move_predictions = Dense(5, activation='softmax')(x)
  rew_predictions = Dense(1, activation='tanh')(x)

  model = Model(inputs=inputs, outputs=(move_predictions, rew_predictions))
  # model = Model(inputs=inputs, outputs=move_predictions)
  model.compile(optimizer='adam',
                loss=['categorical_crossentropy', 'mse'],
                metrics=['accuracy'])
  return model
