import tensorflow as tf


class dnn():
    # 模型参数
    def __init__(self, ):
        self.weight = 1

    # 输入输出参数
    def init_graph(self):
        # dtype：数据类型。常用的是tf.float32, tf.float64等数值类型
        # shape：数据形状。默认是None，就是一维值，也可以是多维（比如[2, 3], [None, 3] 表示列是3，行不定）
        # name：名称
        # 无值的占位变量
        self.x = tf.placeholder(tf.int32, name="x")
        self.y = tf.placeholder(tf.int32, name="y")
        # 两个参数，初始化变量和名字
        # 有初始化值的变量
        self.weight = tf.Variable(10, name="weight")
        self.out = tf.add(tf.summary(self.x, self.weight), y)
        self.out = tf.nn.sigmoid(self.out)
        self.loss = tf.nn.sigmoid(self.out)
        # 全局迭代梯度次数
        self.global_step = tf.Variable(0, trainable=False)
        # 优化器
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        # 查看可以训练的变量
        trainable_params = tf.trainable_variables()
        print(trainable_params)
        # 计算梯度 第一个参数相对于第二个参数的梯度的计算，维数和第二个参数相同，返回的是tensor
        gradients = tf.gradients(self.loss, trainable_params)

        # 在前向传播与反向传播之后，我们会得到每个权重的梯度diff，这时不像通常那样直接使用这些梯度进行权重更新，而是先求所有权重梯度的平方和sumsq_diff，如果sumsq_diff > clip_gradient，则求缩放因子scale_factor = clip_gradient / sumsq_diff。这个scale_factor在(   0, 1)
        # 之间。如果权重梯度的平方和sumsq_diff越大，那缩放因子将越小。
        # 最后将所有的权重梯度乘以这个缩放因子，这时得到的梯度才是最后的梯度信息。
        # 这样就保证了在一次迭代更新中，所有权重的梯度的平方和在一个设定范围以内，这个范围就是clip_gradient.
        # tf.clip_by_global_norm(t_list, clip_norm, use_norm=None, name=None)
        # t_list
        # 是梯度张量， clip_norm
        # 是截取的比率, 这个函数返回截取过的梯度张量和一个所有张量的全局范数。
        clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
        # 更新参数
        # grads_and_vars: 可选变量(gradient, variable)
        # global_step: 在变量已更新后将增加一。
        # name: 返回的操作的可选名称。默认为传递给Optimizer构造函数的名称。
        self.train_op = opt.apply_gradients(
            zip(clip_gradients, trainable_params), global_step=self.global_step)

    def train(self, sess, x, y):
        # 实际会生成一张图，如果有多个线程要生成多个图，需要指定
        # 这里只用了默认图，实际是调用初始化函数，然后feed_dict确认输入参数
        loss, _, step = sess.run([self.loss, self.train_op, self.global_step], feed_dict={
            self.x: x,
            self.y: y
        })
        return loss, step

    def predict(self, sess, feat_index, feat_value):
        result = sess.run([self.out], feed_dict={
            self.x: x,
            self.y: y
        })
        return result

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

if __name__ == '__main__':
    with tf.Session() as sess:
        model = dnn()
        # 初始化所有的全局变量添加到图里
        sess.run(tf.global_variables_initializer)
        # 初始化所有局部变量 添加到图里
        sess.run(tf.local_variables_initializer())
