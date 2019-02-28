import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base

#如果flag 数字为1为训练，其它为预测
tf.app.flags.DEFINE_integer("is_train", 0, "指定是否是训练模型，还是拿数据去预测")
FLAGS = tf.app.flags.FLAGS

def para(shape):
    return tf.Variable(initial_value=tf.random_normal(shape=shape))


def create_model(x):

    with tf.variable_scope("Conv1"):
        # Reshape x
        x_input = tf.reshape(x, shape=[-1, 28, 28, 1])

        # Filter
        weights_conv1 = para(shape=[5, 5, 1, 32])
        bias_conv1 = para(shape=[32])
        x_conv1 = tf.nn.conv2d(x_input, weights_conv1, strides=[1, 1, 1, 1], padding="SAME") + bias_conv1

        x_relu1 = tf.nn.relu(x_conv1)
        x_pool1 = tf.nn.max_pool(x_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    with tf.variable_scope("Conv2"):
        # Filter
        weights_conv2 = para(shape=[5, 5, 32, 64])
        bias_conv2 = para(shape=[64])
        x_conv2 = tf.nn.conv2d(x_pool1, weights_conv2, strides=[1, 1, 1, 1], padding="SAME") + bias_conv2

        x_relu2 = tf.nn.relu(x_conv2)
        x_pool2 = tf.nn.max_pool(x_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


    with tf.variable_scope("FC"):
        x_fc = tf.reshape(x_pool2, shape=[-1, 7*7*64])
        weights_fc = para(shape=[7*7*64, 10])
        bias_fc = para(shape=[10])

        y_predict = tf.matmul(x_fc, weights_fc) + bias_fc

    return y_predict


def cnn_mnist():
    mnist = input_data.read_data_sets("./mnist_data/", one_hot=True)
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_true = tf.placeholder(tf.float32, shape=[None, 10])

    y_predict = create_model(x)

    with tf.variable_scope("softmax_crossentropy"):
        # labels:真实值 [None, 10]  one_hot
        # logits:全脸层的输出[None,10]
        # 返回每个样本的损失组成的列表
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))

    # 4、梯度下降损失优化
    with tf.variable_scope("optimizer"):
        # 学习率
        train_op = tf.train.AdamOptimizer(0.001).minimize(loss)

    # 5、得出每次训练的准确率（通过真实值和预测值进行位置比较，每个样本都比较）
    with tf.variable_scope("accuracy"):
        equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # （2）收集要显示的变量
    # 先收集损失和准确率
    tf.summary.scalar("losses", loss)
    tf.summary.scalar("acc", accuracy)

    # 初始化变量op
    init_op = tf.global_variables_initializer()

    # （3）合并所有变量op
    merged = tf.summary.merge_all()

    # 创建模型保存和加载
    saver = tf.train.Saver()

    # 开启会话去训练
    with tf.Session() as sess:
        # 初始化变量
        sess.run(init_op)

        # （1）创建一个events文件实例
        file_writer = tf.summary.FileWriter("./tmp/summary/", graph=sess.graph)

        # 加载模型
        if os.path.exists("./tmp/modelckpt/checkpoint"):
            saver.restore(sess, "./tmp/modelckpt/fc_nn_model")

        if FLAGS.is_train == 1:
            # 循环步数去训练
            for i in range(3000):
                # 获取数据，实时提供
                # 每步提供50个样本训练
                mnist_x, mnist_y = mnist.train.next_batch(50)
                # 运行训练op
                sess.run(train_op, feed_dict={x: mnist_x, y_true: mnist_y})
                print("训练第%d步的准确率为：%f, 损失为：%f " % (i + 1,
                                                   sess.run(accuracy, feed_dict={x: mnist_x, y_true: mnist_y}),
                                                   sess.run(loss, feed_dict={x: mnist_x, y_true: mnist_y})
                                                   )
                      )

                # 运行合变量op，写入事件文件当中
                summary = sess.run(merged, feed_dict={x: mnist_x, y_true: mnist_y})
                file_writer.add_summary(summary, i)
                if i % 100 == 0:
                    saver.save(sess, "./tmp/modelckpt/fc_nn_model")

        else:
            num = 0
            # 如果不是训练，我们就去进行预测测试集数据
            for i in range(100):
                # 每次拿一个样本预测
                mnist_x, mnist_y = mnist.test.next_batch(1)
                print("第%d个样本的真实值为：%d, 模型预测结果为：%d" % (
                    i + 1,
                    tf.argmax(sess.run(y_true, feed_dict={x: mnist_x, y_true: mnist_y}), 1).eval(),
                    tf.argmax(sess.run(y_predict, feed_dict={x: mnist_x, y_true: mnist_y}), 1).eval()
                )
                )
                # Calculate accuracy
                if tf.argmax(sess.run(y_true, feed_dict={x: mnist_x, y_true: mnist_y}), 1).eval() == tf.argmax(
                        sess.run(y_predict, feed_dict={x: mnist_x, y_true: mnist_y}), 1).eval():
                    num += 1

            print(num / 100)

    return None


if __name__ == "__main__":
    print()
    cnn_mnist()
