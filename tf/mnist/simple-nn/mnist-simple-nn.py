import argparse
import sys
import tensorflow as tf
from   tensorflow.examples.tutorials.mnist import input_data as mnist_data

ParsedArgs = None

def simplenn(_):
    print('MNIST Simple NN')
    # one_hot vectors are vectors that are 0 in most dimension and 1 in one dimension
    # for e.g. our label vector for:
    # 3 would be: { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0} 
    # 6 would be: { 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0}
    # etc
    # So labels is a tensor of [55000, 10] - 55000 entries, each entry a one-hot vector with 10 elements
    mnistDataSet = mnist_data.read_data_sets(ParsedArgs.mnistInputDataDir, one_hot=True)

    # Create the model
    images         = tf.placeholder(tf.float32, [None, 784]) # placeholder for images
    weights        = tf.Variable(tf.zeros([784, 10]))        # 784 weights per class - 10 classes total
    bias           = tf.Variable(tf.zeros([10]))             # one bias per class - 10 classes total  
    computedLabels = tf.matmul(images, weights) + bias       # y = f(x) = Wx + b : Score function for each class with input image (x)

    # Expected labels (one-hot-vectors) for 'None' images ('None' = actual number to be given later)
    # When we extract the training images from the MNIST data, we'll also extract the corresponding labels
    expectedLabels = tf.placeholder(tf.float32, [None, 10])

    # Loss & Loss Optimizer Function
    crossEntropy   = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=expectedLabels, logits=computedLabels))
    train_step     = tf.train.GradientDescentOptimizer(0.5).minimize(crossEntropy)

    session        = tf.InteractiveSession()

    tf.global_variables_initializer().run()
 
    # train
    for _ in range(3000):
        # using small batches of random data is stochastic training - in this case stochastic gradient descent
        trainingImages, trainingLabels = mnistDataSet.train.next_batch(100)
        session.run(train_step, feed_dict = {images: trainingImages, expectedLabels: trainingLabels})

    # coming here means our weights/biases would have been learnt already

    # test the trained model
    prediction = tf.equal(tf.argmax(computedLabels, 1), tf.argmax(expectedLabels, 1))
    accuracy   = tf.reduce_mean(tf.cast(prediction, tf.float32))

    # trigger the testing of the model with the test images from the data set
    print(session.run(accuracy, feed_dict = {images: mnistDataSet.test.images, expectedLabels: mnistDataSet.test.labels}))
    print('Program End')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mnistInputDataDir',
                        type=str,
                        default='/users/xxx/mldl/mnistinputdata',
                        help='Directory for mnist image/label data')

    ParsedArgs, UnParsedArgs = parser.parse_known_args()
    tf.app.run(main=simplenn, argv=[sys.argv[0]] + UnParsedArgs)