import tensorflow as tf

class NetworkModule:

    def __init__(self, saver, session):
        self.saver = saver
        self.sess = session
        self.x = tf.placeholder(tf.float32,[None,8*8*3])
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

        self.y_target = tf.placeholder(tf.float32,[None,M*M])
        self.value_target = tf.placeholder(tf.float32,[None,1])
        self.train_target = tf.placeholder(tf.float32,[None,M*M])

        self.true_loss = tf.nn.l2_loss(y_target-probs) + tf.nn.l2_loss(value-value_target)
        self.regularizer = tf.reduce_sum([tf.nn.l2_loss(x)for x in tf.global_variables()])*0.0001
        loss = true_loss + self.regularizer

        opt = tf.train.AdamOptimizer(0.0003)
        
        self.train_opt = opt.minimize(loss)
    
    def initialize_variables(self):
        sess.run(tf.global_variables_initializer())
    
    def load(self, filename):
        self.saver.restore(self.sess, filename)
    
    def save(self, filename):
        self.saver.save(self.sess, filename)
    
    def run(self, batch):
        return self.sess.run([self.probs,self.value],
                      feed_dict={self.x:batch})
    
    def train(self, batch, values, target)
        _,true_loss,regularizer =  self.sess.run([self.train_opt,self.true_loss,self.regularizer],
                                feed_dict={self.x:batch,self.value_target:values, self.y_target:target})
        return true_loss,regularizer
    
    def run_net(board, masks,net):
    data, value = sess.run(joint,feed_dict={x:np.reshape(np.concatenate([convert(board),masks],2),[1,8*8*3])})
    return np.reshape(data,[M,M]), value
