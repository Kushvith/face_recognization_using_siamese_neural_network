# custom L1 Distance Layer

import tensorflow as tf
from tensorflow.keras.layers import Layer 

class L1Dist(Layer):
  def __init__(self,**kwargs):
    super().__init__()
    # Magic haooens Here
  def call(self,input_embedding,validation_embedding):
    return tf.math.abs(input_embedding - validation_embedding)

