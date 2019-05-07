import tensorflow as tf

class NetworkModule:

    def __init__(self, saver, session):
        self.saver = saver
        self.sess = session
        self.x = tf.placeholder(tf.float32,[None,M*M*3])
        input_layer = tf.reshape(x, [-1,8,8,3])

        result = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu)(input_layer)

        for i in range(6):
            result = tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu)(result)


        conv_reduce = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu)(result)

        conv_reduce2 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[3, 3],
            padding="valid",
            activation=tf.nn.relu)(conv_reduce)

        conv_reduce3 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[3, 3],
            padding="valid",
            activation=tf.nn.relu)(conv_reduce2)

        conv_reduce4 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[3, 3],
            padding="valid",
            activation=tf.nn.relu)(conv_reduce3)


        conv_probs = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu)(result)

        self.probs = tf.nn.softmax(tf.keras.layers.Flatten()(conv_probs))

        intermediate = tf.keras.layers.Dense(256,activation=tf.nn.relu)(tf.keras.layers.Flatten()(conv_reduce4))
        self.value = tf.keras.layers.Dense(1,activation=tf.nn.sigmoid)(intermediate)

        joint = tf.tuple([probs,value])

        y_target = tf.placeholder(tf.float32,[None,M*M])
        value_target = tf.placeholder(tf.float32,[None,1])
        train_target = tf.placeholder(tf.float32,[None,M*M])

        true_loss = tf.nn.l2_loss(y_target-probs) + tf.nn.l2_loss(value-value_target)

        self.loss = true_loss + tf.reduce_sum([tf.nn.l2_loss(x)for x in tf.global_variables()])*0.0001

        opt = tf.train.AdamOptimizer(0.0003)
        
        self.train_opt = opt.minimize(loss)
