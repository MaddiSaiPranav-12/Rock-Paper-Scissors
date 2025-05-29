import tensorflow as tf
layers = tf.keras.layers

class Voxel_flow_model(tf.keras.Model):
    def __init__(self):
        super(Voxel_flow_model, self).__init__()

        # Define convolutional layers using Keras layers
        self.conv1 = layers.Conv2D(64, 5, strides=1, padding='same', activation='relu',
                                   kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01))
        self.conv2 = layers.Conv2D(128, 5, strides=2, padding='same', activation='relu',
                                   kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01))
        self.conv3 = layers.Conv2D(256, 5, strides=2, padding='same', activation='relu',
                                   kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01))
        self.conv4 = layers.Conv2D(512, 3, strides=2, padding='same', activation='relu',
                                   kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01))

        self.deconv1 = layers.Conv2DTranspose(256, 4, strides=2, padding='same', activation='relu',
                                              kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01))
        self.deconv2 = layers.Conv2DTranspose(128, 4, strides=2, padding='same', activation='relu',
                                              kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01))
        self.deconv3 = layers.Conv2DTranspose(64, 4, strides=2, padding='same', activation='relu',
                                              kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01))

        # Final output layer
        self.final_conv = layers.Conv2D(3, 3, strides=1, padding='same', activation='sigmoid',
                                        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01))

    def call(self, x, training=False):
        # Encoder
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        # Decoder
        d1 = self.deconv1(x4)
        d2 = self.deconv2(d1)
        d3 = self.deconv3(d2)

        # Final prediction
        out = self.final_conv(d3)

        return out, None  # Returning dummy second output to match original API

    def inference(self, input_images):
        return self.call(input_images, training=False)

    def train_step(self, input_images, target_images, optimizer):
        with tf.GradientTape() as tape:
            pred, _ = self.call(input_images, training=True)
            loss = tf.reduce_mean(tf.abs(pred - target_images))

        grads = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return loss
