
#-*- coding: utf-8 -*-
"""
Created on Wed Aug 1 08:50:40 2018
class MyGan
@author: Dujin
"""
runfile('C:/users/dujin/desktop/tensorflow/infogan/basefunc.py')

class MyGan(object):
    def __init__(self,sess,data_path,ydim = 12 ,zdim = 10,batch_size = 64, input_height = 28 , input_width =28,channel = 1,epoch = 10000):
        self.sess = sess
        self.data_path = 'C:/users/dujin/desktop/tensorflow/infogan/data/'+'mnist'
        self.ydim = ydim
        self.zdim = zdim
        self.batch_size = batch_size
        self.input_height = input_height
        self.input_width = input_width
        self.cdim = channel
        self.data_X , self.data_y = load_mnist(self.data_path)
        self.beta1 = 0.5
        self.learning_rate = 0.0002
        self.len_continuous_code = 2
        self.len_discrete_code = 10
        self.num_batches = len(self.data_X) // self.batch_size
        self.sample_num = 64
        self.epoch = epoch

    def generator(self, z, y, is_training=True, reuse=False):
        # [batch_size, z_dim+y_dim] > [batch_size, 1024] > [batch_size, 128*7*7] > 
        # [batch_size, 7, 7, 128] > [batch_size, 14, 14, 64] > [batch_size, 28, 28, 1]
        with tf.variable_scope("generator", reuse=reuse):

            # merge noise and code
            z = tf.concat([z, y], 1)

            net = tf.nn.relu(bn(linear(z, 1024, scope='g_fc1'), is_training=is_training, scope='g_bn1'))
            net = tf.nn.relu(bn(linear(net, 128 * 7 * 7, scope='g_fc2'), is_training=is_training, scope='g_bn2'))
            net = tf.reshape(net, [self.batch_size, 7, 7, 128])
            net = tf.nn.relu(
                bn(deconv2d(net, [self.batch_size, 14, 14, 64], 4, 4, 2, 2, name='g_dc3'), is_training=is_training,
                   scope='g_bn3'))

            out = tf.nn.sigmoid(deconv2d(net, [self.batch_size, 28, 28, 1], 4, 4, 2, 2, name='g_dc4'))

            return out

    def discriminatior(self, x, is_training=True, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):
    
            net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='d_conv1'))
            net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='d_conv2'), is_training=is_training, scope='d_bn2'))
            net = tf.reshape(net, [self.batch_size, -1])
            net = lrelu(bn(linear(net, 1024, scope='d_fc3'), is_training=is_training, scope='d_bn3'))
            out_logit = linear(net, 1, scope='d_fc4')
            out = tf.nn.sigmoid(out_logit)
    
            return out, out_logit, net
    def classifier(self, x, is_training=True, reuse=False):
        # x from discirminator net
        # [batch_size, 64] > [batch_size, y_dim]
        with tf.variable_scope("classifier", reuse=reuse):

            net = lrelu(bn(linear(x, 64, scope='c_fc1'), is_training=is_training, scope='c_bn1'))
            out_logit = linear(net, self.ydim, scope='c_fc2')
            out = tf.nn.softmax(out_logit)

            return out, out_logit
            
    def bulid_model(self):
        self.z = tf.placeholder(dtype = tf.float32 , shape = [self.batch_size , self.zdim],name = 'z')
        self.y = tf.placeholder(tf.float32,shape = [self.batch_size , self.ydim],name = 'y')
        self.dic_input = tf.placeholder(tf.float32, shape = [self.batch_size, self.input_height, self.input_width,self.cdim] , name = 'dic_input')
        # cal  loss
        # real image loss
        D_real , D_real_logits, _ = self.discriminatior(self.dic_input,is_training= True)
        # fake image loss
        G = self.generator(self.z,self.y,is_training= True , reuse = False)
        D_fake , D_fake_logits, input_classifier = self.discriminatior(G, is_training = True , reuse = True)
        #get loss for Discriminator
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits= D_real_logits,labels =  tf.ones_like(D_real_logits)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits= D_fake_logits,labels =  tf.zeros_like(D_fake_logits)))
        self.d_loss = d_loss_fake + d_loss_real
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake)))

        ## 2. Information Loss" 
        code_fake, code_logit_fake = self.classifier(input_classifier, is_training=True, reuse=False)
        
        disc_code_est = code_logit_fake[:, :self.len_discrete_code]
        disc_code_tg = self.y[:, :self.len_discrete_code]
        q_disc_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=disc_code_est, labels=disc_code_tg))
        
        cont_code_est = code_logit_fake[:, self.len_discrete_code:]
        cont_code_tg = self.y[:, self.len_discrete_code:]
        q_cont_loss = tf.reduce_mean(tf.reduce_sum(tf.square(cont_code_tg - cont_code_est), axis=1))
        self.q_loss = q_disc_loss + q_cont_loss

        self.fake_images = self.generator(self.z, self.y, is_training=False, reuse=True)

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]
        q_vars = [var for var in t_vars if ('d_' in var.name) or ('c_' in var.name) or ('g_' in var.name)]

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.d_loss, var_list=d_vars)
            self.g_optim = tf.train.AdamOptimizer(self.learning_rate * 5, beta1=self.beta1) \
                .minimize(self.g_loss, var_list=g_vars)
            self.q_optim = tf.train.AdamOptimizer(self.learning_rate * 5, beta1=self.beta1) \
                .minimize(self.q_loss, var_list=q_vars)

    def train(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        for epoch in range(0, self.epoch):
            for idx in range(0, self.num_batches):
                batch_images = self.data_X[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_labels = self.data_y[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_codes = np.concatenate((batch_labels, np.random.uniform(-1, 1, size=(self.batch_size, 2))),
                                 axis=1)
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.zdim]).astype(np.float32)  
                         
                _, d_loss = self.sess.run([self.d_optim,self.d_loss],feed_dict={self.dic_input: batch_images, self.y: batch_codes,self.z: batch_z})
                
                self.sess.run([self.g_optim, self.q_optim], feed_dict={self.dic_input: batch_images, self.z: batch_z, self.y: batch_codes})
                print(d_loss)
                

    
    def visualize_result(self):
        """ random noise, random discrete code, fixed continuous code """
        image_frame_dim = 8
        y = np.random.choice(self.len_discrete_code, self.batch_size)
        y_one_hot = np.zeros((self.batch_size, self.ydim))
        y_one_hot[np.arange(self.batch_size), y] = 1
        z_sample = np.random.uniform(-1, 1, size=(self.batch_size, self.zdim))
        self.samples1 = self.sess.run(self.fake_images, feed_dict={self.z: z_sample, self.y: y_one_hot})
        save_images(self.samples1[:64, :, :, :],[8,8],'r_d_code'+'.png')
        """ random noise  specified discrete code, fixed continuous code """
        n_styles = 10  
        np.random.seed()
        si = np.random.choice(self.batch_size, n_styles)
        for l in range(self.len_discrete_code):
            y = np.zeros(self.batch_size, dtype=np.int64) + l
            y_one_hot = np.zeros((self.batch_size, self.ydim))
            y_one_hot[np.arange(self.batch_size), y] = 1
        self.samples2 = self.sess.run(self.fake_images, feed_dict={self.z: z_sample, self.y: y_one_hot})
        save_images(self.samples2[:64, :, :, :],[8,8],'s_d_code'+'.png')
        """ fixed noise  pecified discrete code, gradual change continuous code  """
        assert self.len_continuous_code == 2
        c1 = np.linspace(-1, 1, image_frame_dim)
        c2 = np.linspace(-1, 1, image_frame_dim)
        xv, yv = np.meshgrid(c1, c2)
        xv = xv[:image_frame_dim,:image_frame_dim]
        yv = yv[:image_frame_dim, :image_frame_dim]

        c1 = xv.flatten()
        c2 = yv.flatten()

        z_fixed = np.zeros([self.batch_size, self.zdim])

        for l in range(self.len_discrete_code):
            y = np.zeros(self.batch_size, dtype=np.int64) + l
            y_one_hot = np.zeros((self.batch_size, self.ydim))
            y_one_hot[np.arange(self.batch_size), y] = 1

            y_one_hot[np.arange(image_frame_dim*image_frame_dim), self.len_discrete_code] = c1
            y_one_hot[np.arange(image_frame_dim*image_frame_dim), self.len_discrete_code+1] = c2

            self.samples3 = self.sess.run(self.fake_images,
                                    feed_dict={ self.z: z_fixed, self.y: y_one_hot})
            save_images(self.samples3[:64, :, :, :],[8,8],str(l)+'.png')
        
        
infoGAN = MyGan(tf.Session(),'mnist',epoch = 10)
infoGAN.bulid_model()
infoGAN.train()
infoGAN.visualize_result()
