import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, Concatenate, Flatten, Conv2D
from keras.models import Model

def build_model(grid_world_shape, Pi_shape=5):
  inputs = Input(shape=grid_world_shape)
  x0 = Conv2D(32, (2, 2), padding="same", activation="relu")(inputs)
  x0 = Conv2D(8, (2, 2), padding="same", activation="relu")(x0)
  x0 = Flatten()(x0)
  x = Dense(32, activation='relu')(x0)
  x = Dense(16, activation='relu')(x)
  move_predictions = Dense(Pi_shape, activation='softmax')(x)
  value_predictions = Dense(1, activation='tanh')(x)

  model = Model(inputs=inputs, outputs=(move_predictions, value_predictions))
  # model = Model(inputs=inputs, outputs=move_predictions)
  model.compile(optimizer='adam',
                loss=['categorical_crossentropy', 'mse'],
                metrics=['accuracy'])
  return model

def build_policy_model(grid_world_shape, Pi_shape):
  inputs = Input(shape=grid_world_shape)
  x0 = Flatten()(inputs)
  x = Dense(32, activation='relu')(x0)
  # x = Dense(32, activation='relu')(x)
  x = Dense(16, activation='relu')(x)
  move_predictions = Dense(tf.keras.backend.prod(Pi_shape), activation='softmax')(x)
  policy = keras.layers.Reshape(Pi_shape)(move_predictions)

  model = Model(inputs=inputs, outputs=(policy))
  # model = Model(inputs=inputs, outputs=move_predictions)
  model.compile(optimizer='adam',
                loss=['categorical_crossentropy'],
                metrics=['accuracy'])
  return model