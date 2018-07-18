import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def quantizeData(data):
    ## you can design your own quantization method
    ## the following method is a naive implementation and it quantizes the numbers to -1 or +1.
    quantizedData = np.sign(data)
    return quantizedData

def quantization_training():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # Python optimisation variables
    learning_rate = 0.5
    epochs = 10 #10
    batch_size = 100

    # declare the training data placeholders
    # input x - for 28 x 28 pixels = 784
    x = tf.placeholder(tf.float32, [None, 784], name='x')
    # now declare the output data placeholder - 10 digits
    y = tf.placeholder(tf.float32, [None, 10], name='y')

    # now declare the weights connecting the input to the hidden layer
    W1 = tf.Variable(tf.random_normal([784, 300], stddev=0.03), name='W1')
    b1 = tf.Variable(tf.random_normal([300]), name='b1')
    # and the weights connecting the hidden layer to the output layer
    W2 = tf.Variable(tf.random_normal([300, 10], stddev=0.03), name='W2')
    b2 = tf.Variable(tf.random_normal([10]), name='b2')

    # calculate the output of the hidden layer
    hidden_out = tf.add(tf.matmul(x, W1), b1)
    hidden_out = tf.nn.relu(hidden_out)

    # now calculate the hidden layer output - in this case, let's use a softmax activated
    # output layer
    y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))

    # now let's define the cost function which we are going to train the model on
    y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
    cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)
                                                  + (1 - y) * tf.log(1 - y_clipped), axis=1))

    # add an optimiser
    optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    # finally setup the initialisation operator
    init_op = tf.global_variables_initializer()

    # define an accuracy assessment operation
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    # create a saver
    saver = tf.train.Saver()
    sess = tf.Session()

    # start the session
    # initialise the variables
    print("General Training")
    sess.run(init_op)
    total_batch = int(len(mnist.train.labels) / batch_size)
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
            _, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))

    # save the trained model
    saver.save(sess, "C:\\Users\\xiaowei\\PycharmProjects\\testTF\\my_test_model")
    print("General Training with 32 bits floating point precision complete!")
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

## quantize a part of all the parameters
    print("\nQuantize W1!")
    variable_name = [v.name for v in tf.trainable_variables()]
    weight_1 = sess.run(variable_name[0])
    weight_1_quantized = quantizeData(weight_1)
    assign_op_w1 = W1.assign(weight_1_quantized)
    sess.run(assign_op_w1)
    # check the result, you can uncomment the following for debug
    #values = sess.run(variable_name)
    #for k, v in zip(variable_name, values):
    #    print(k, v)

## re-training
    print("\nRe-training with freezed W1!")
    total_batch = int(len(mnist.train.labels) / batch_size)
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
            # we need to assign the quantized value to freeze it during training
            sess.run(assign_op_w1)
            # training
            _, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
    sess.run(assign_op_w1)

## quantization another part
    print("\nQuantize W2!")
    weight_2 = sess.run(variable_name[2])
    weight_2_quantized = quantizeData(weight_2)
    assign_op_w2 = W2.assign(weight_2_quantized)
    sess.run(assign_op_w2)

## re-training
    print("\nRe-training with freezed W1 and W2!")
    total_batch = int(len(mnist.train.labels) / batch_size)
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
            # we need to assign the quantized value to freeze it during training
            sess.run(assign_op_w1)
            sess.run(assign_op_w2)
            # training
            _, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
    sess.run(assign_op_w1)
    sess.run(assign_op_w2)

## Quantize the rest
    print("\nQuantize the rest!")
    bias_1 = sess.run(variable_name[1])
    bias_1_quantized = quantizeData(bias_1)
    assign_op_b1 = b1.assign(bias_1_quantized)
    sess.run(assign_op_b1)
    bias_2 = sess.run(variable_name[3])
    bias_2_quantized = quantizeData(bias_2)
    assign_op_b2 = b2.assign(bias_2_quantized)
    sess.run(assign_op_b2)

## incremental training completes, and save the model
    # save the trained model
    print("\nRe-Training complete!")
    saver.save(sess, "C:\\Users\\xiaowei\\PycharmProjects\\testTF\\my_test_model")
    #writer.add_graph(sess.graph)
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

## check the result, you can uncomment the following for debug
    values = sess.run(variable_name)
    for k, v in zip(variable_name, values):
        print(k, v)

def mnist_restore():

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    sess = tf.Session()
    saver = tf.train.import_meta_graph('C:\\Users\\xiaowei\\PycharmProjects\\testTF\\my_test_model.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    y = graph.get_tensor_by_name("y:0")
    accuracy = graph.get_tensor_by_name('accuracy:0')
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

if __name__ == "__main__":

    #mnist_restore()
    quantization_training()