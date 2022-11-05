import logging
import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import tensorflow.keras.layers as layers
from scipy.stats import multivariate_normal as normal
DELTA_CLIP = 50.0


class BSDESolver(object):
    """The fully connected neural network model."""
    def __init__(self, config, bsde):
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.bsde = bsde
       
        self.model = NonsharedModel(config, bsde)

        try:
            lr_schedule = config.net_config.lr_schedule
        except AttributeError:
            lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                self.net_config.lr_boundaries, self.net_config.lr_values)     
            
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-8)

    def train(self):
        start_time = time.time()
        training_history = []
        valid_data = self.bsde.sample(self.net_config.valid_size)

        # begin sgd iteration
        for step in tqdm(range(self.net_config.num_iterations+1)):
            if step % self.net_config.logging_frequency == 0:
                loss = self.loss_fn(valid_data, training=False).numpy()
                y_init = self.model.simulate_path(valid_data)[0,0,0] # or y_init = self.model.subnet[0].call(valid_data[0][:,:,0],False)[0].numpy()[0]
                elapsed_time = time.time() - start_time
                training_history.append([step, loss, y_init, elapsed_time])
                if self.net_config.verbose:
                    print("step: %5u,    loss: %.4e, Y0: %.4e,   elapsed time: %3u" % (
                        step, loss, y_init, elapsed_time))
            self.train_step(self.bsde.sample(self.net_config.batch_size))            
        return np.array(training_history)
   
    def loss_fn(self, inputs, training):

        x, Poisson, jumps, dw = inputs
        y_terminal, penalty = self.model(inputs, training)
        
        delta = y_terminal - self.bsde.g_tf(self.bsde.total_time, x[:, :, -1])
        # use linear approximation outside the clipped range
        loss = tf.reduce_mean(tf.where(tf.abs(delta) < DELTA_CLIP, tf.square(delta),
                                    2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP ** 2))

        return loss + penalty

    def grad(self, inputs, training):
        with tf.GradientTape(persistent=True) as tape:
            loss = self.loss_fn(inputs, training)
            
        grad = tape.gradient(loss, self.model.trainable_variables)
        del tape
        return grad

    @tf.function
    def train_step(self, train_data):
        grad = self.grad(train_data, training=True)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))     


class NonsharedModel(tf.keras.Model):
    def __init__(self, config, bsde):
        super(NonsharedModel, self).__init__()
        self.config = config
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.bsde = bsde       
        self.dim = bsde.dim       
        self.subnet = [FeedForwardSubNet(config,bsde.dim) for _ in range(self.bsde.num_time_interval)]
        self.subnetCompensator = [FeedForwardSubNet(config,bsde.dim) for _ in range(self.bsde.num_time_interval)]

    @tf.function  
    def call(self, inputs, training):
        x, Poisson, jumps, dw = inputs
        time_stamp = np.arange(0, self.eqn_config.num_time_interval) * self.bsde.delta_t
        all_one_vec = tf.ones(shape=tf.stack([tf.shape(Poisson)[0], 1]), dtype=self.net_config.dtype)
        y = all_one_vec * self.subnet[0].call(all_one_vec * self.bsde.x_init,training)
        
        penalty = 0.0
        
        for t in range(0, self.bsde.num_time_interval):
                
            u = self.subnet[t].call(x[:, :, t ]* tf.math.exp(jumps[:, :, t]),training=True) - self.subnet[t].call(x[:, :, t],training=True)   
            
            z = self.subnet[t].grad(x[:, :, t]) * self.bsde.getFsdeDiffusion(t,x[:, :, t])/ self.bsde.dim
    
            compensator = self.subnetCompensator[t].call(x[:, :, t],training=True)
            compensatedJump = u-compensator
            penalty = penalty + (tf.reduce_mean(tf.square(compensatedJump)))
            
            y = y - self.bsde.delta_t * (
                self.bsde.f_tf(time_stamp[t], x[:, :, t], y, z)) + compensatedJump + \
               tf.reduce_sum(z * self.bsde.getFsdeDiffusion(time_stamp[t],x[:, :, t]) * dw[:, :, t], 1, keepdims=True) 

        return y, penalty
         
    @tf.function
    def predict_step(self, data):
        x, Poisson, jumps, dw = data[0] 
        
        time_stamp = np.arange(0, self.eqn_config.num_time_interval) * self.bsde.delta_t
        all_one_vec = tf.ones(shape=tf.stack([tf.shape(Poisson)[0], 1]), dtype=self.net_config.dtype)
        
        y = all_one_vec * self.subnet[0].call(all_one_vec * self.bsde.x_init,False)
           
       
        history = tf.TensorArray(self.net_config.dtype,size=self.bsde.num_time_interval+1)     
        history = history.write(0,y)
        
        
        for t in range(0, self.bsde.num_time_interval):
                
            u = self.subnet[t].call(x[:, :, t ]* tf.math.exp(jumps[:, :, t]),training=False) - self.subnet[t].call(x[:, :, t],training=False)  

            z = self.subnet[t].grad(x[:, :, t]) * self.bsde.getFsdeDiffusion(t,x[:, :, t])/ self.bsde.dim

            compensator = self.subnetCompensator[t].call(x[:, :, t],training=False)       
            compensatedJump = u-compensator

            y = y - self.bsde.delta_t * (
                self.bsde.f_tf(time_stamp[t], x[:, :, t], y, z)) + compensatedJump + \
               tf.reduce_sum(z * self.bsde.getFsdeDiffusion(time_stamp[t],x[:, :, t]) * dw[:, :, t], 1, keepdims=True) 
               
            history = history.write(t+1,y)

        history = tf.transpose(history.stack(),perm=[1,2,0])
        return Poisson, jumps, x, history, dw     

    def simulate_path(self,num_sample):
        return self.predict(num_sample)[3]          


class FeedForwardSubNet(tf.keras.Model):
    def __init__(self, config,dim):
        super(FeedForwardSubNet, self).__init__()        
        num_hiddens = config.net_config.num_hiddens
        self.bn_layers = [
            tf.keras.layers.BatchNormalization(
                momentum=0.99,
                epsilon=1e-6,
                beta_initializer=tf.random_normal_initializer(0.0, stddev=0.1),
                gamma_initializer=tf.random_uniform_initializer(0.1, 0.5)
            )
            for _ in range(len(num_hiddens) + 2)]
        self.dense_layers = [tf.keras.layers.Dense(num_hiddens[i],
                                                   use_bias=False,
                                                   activation=None,)
                             for i in range(len(num_hiddens))]
        self.dense_layers.append(tf.keras.layers.Dense(dim, activation=None))

    def call(self, x, training):
        """structure: bn -> (dense -> bn -> relu) * len(num_hiddens) -> dense """
        x = self.bn_layers[0](x, training)
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            x = self.bn_layers[i+1](x, training)
            # x = tf.nn.relu(x)
            x = tf.nn.sigmoid(x)
        x = self.dense_layers[-1](x)
        
        """
        Questa somma serve per ridurre la dimensione da d a 1 ora la rete
        rappresenta una funzione da Rd a R.
        """
        tf.math.reduce_sum(x,0)
        return x

    def grad(self, x):
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float64)
        with tf.GradientTape(watch_accessed_variables=True) as t:
            t.watch(x_tensor)
            out = self.call(x_tensor,training=False)
        grad = t.gradient(out,x_tensor)
        del t
        return grad


    

