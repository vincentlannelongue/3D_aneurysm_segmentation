import tensorflow as tf

class DiceLoss(tf.keras.losses.Loss):
  
    def __init__(self, smooth=1e-6, gama=2, name=None, **kwargs):
        """
        Dice Coefficient implementation adapted from 
        https://dev.to/_aadidev/3-common-loss-functions-for-image-segmentation-545o.
        Calculates the similarity between images, and is similar to the 
        Intersection over Union heuristic.
        A common criticism is the nature of its resulting search space, which is 
        non-convex, several modifications have been made to make the Dice Loss 
        more tractable for solving using methods such as L-BFGS and Stochastic 
        Gradient Descent.

        Arguments:
            smooth: 
            gama: 
        """
        super(DiceLoss, self).__init__(name=name)
        self.name = 'DiceLoss'
        self.smooth = smooth
        self.gama = gama
        #super(DiceLoss, self).__init__(**kwargs)
    
    def get_config(self):
      config = super(DiceLoss, self).get_config()
      config.update({"DiceLoss" : self.name})
      return config

    def call(self, y_true, y_pred):
        y_true, y_pred = tf.cast(
            y_true, dtype=tf.float32), tf.cast(y_pred, tf.float32)
        nominator = 2 * \
            tf.reduce_sum(tf.multiply(y_pred, y_true)) + self.smooth
        denominator = tf.reduce_sum(
            y_pred ** self.gama) + tf.reduce_sum(y_true ** self.gama) + self.smooth
        result = 1 - tf.divide(nominator, denominator)
        return result