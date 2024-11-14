import logging
import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import LeakyReLU
from scipy.stats import multivariate_normal as normal
import os
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
            
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule, epsilon=1e-8)

    def train(self):
        start_time = time.time()
        training_history = []
        valid_data = self.bsde.sample(self.net_config.valid_size)

        # begin batch iteration
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
        
        loss = tf.reduce_mean( delta**2 )

        return 10 * loss + penalty

    def grad(self, inputs, training):
        with tf.GradientTape(persistent=True) as tape:
            loss = self.loss_fn(inputs, training)
            
        grad = tape.gradient(loss, self.model.trainable_variables)
        #clipped_gradients = [tf.clip_by_value(g, -5., 5.) for g in grad]
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
        self.subnet = [FeedForwardSubNet(config,1) for _ in range(self.bsde.num_time_interval)]
        self.subnetControl = [FeedForwardSubNetControl(config,bsde.dim) for _ in range(self.bsde.num_time_interval)]
        self.subnetCompensator = [FeedForwardSubNet(config,1) for _ in range(self.bsde.num_time_interval)]
        self.y_init = tf.Variable(np.random.uniform(low = self.net_config.y_init_range[0], high = self.net_config.y_init_range[1], size=[1]),dtype=self.net_config.dtype)
        
        
    def save_model(self, path_dir = 'Model_LR'):
        
        try: 
            os.mkdir(path_dir)
 
            nets = self.subnet
            netsControl = self.subnetControl
            netsComp = self.subnetCompensator
        
            for i in range(self.bsde.num_time_interval-1):
                path = path_dir + '/net_{}'.format(i)
                pathComp = path_dir + '/netComp_{}'.format(i)
                model = nets[i]
                model.save(path)
                modelComp = netsComp[i]
                modelComp.save(pathComp)

            print('Directory ', '\033[1m' + path_dir + '\033[0m', ' created')
        
        except OSError:
            print('Directory already existing: try another name')
        
 
    def load_model(self, path_dir = 'Model_LR'):

        nets = [] 
        netsControl = []
        netsComp = []
        for i in range(self.bsde.num_time_interval-1):
            path = path_dir + '/net_{}'.format(i)
            model = tf.keras.models.load_model(path)
            nets.append(model)
            
            pathControl = path_dir + '/netControl_{}'.format(i)
            modelControl = tf.keras.models.load_model(pathControl)
            netsControl.append(model)
            
            
            pathComp = path_dir + '/netComp_{}'.format(i)
            modelComp = tf.keras.models.load_model(pathComp)
            netsComp.append(modelComp)
        
        self.subnet = nets
        self.subnetCompensator = netsComp
        self.subnetControl = netsControl
        

    @tf.function  
    def call(self, inputs, training):
        print('Training!')
        x, Poisson, jumps, dw = inputs
        time_stamp = np.arange(0, self.eqn_config.num_time_interval) * self.bsde.delta_t
        all_one_vec = tf.ones(shape=tf.stack([tf.shape(Poisson)[0], 1]), dtype=self.net_config.dtype)
        
        
        y = all_one_vec * self.y_init
        
        #y = all_one_vec * self.subnet[0].call(all_one_vec * self.bsde.x_init,training)
        
        penalty = 0.0
        one_net = False
        for t in range(0, self.bsde.num_time_interval):
            if one_net:
                batch_size = tf.shape(x)[0]
                t_vector = tf.cast(tf.fill([batch_size, 1], self.bsde.delta_t * t), dtype=x.dtype)
                
                input_concat_z = tf.concat([t_vector, x[:,:,t]], axis=1)
                z = self.subnetControl[0](input_concat_z, training=True)# * self.bsde.getFsdeDiffusion(t,x[:, :, t])/ self.bsde.dim
                
                #input_concat_jump =  tf.concat([t_vector, x[:,:,t], x[:, :, t ] * tf.math.exp(jumps[:, :, t])], axis=1)
                input_concat_jump =  tf.concat([t_vector, x[:,:,t]], axis=1)
                f_Jump = self.subnetCompensator[0].call(input_concat_jump, training=True) * ( x[:, :, t  ] * tf.math.exp(jumps[:, :, t]) - x[:, :, t ] ) 
                
                input_concat_comp = tf.concat([t_vector, x[:,:,t]], axis=1)
                f_compensator = self.subnet[0].call(input_concat_comp, training=True)  
                
                ###
                #f_Jump = self.subnetCompensator[0].call(x[:,:,t],training=True) * tf.reduce_sum( x[:, :, t  ] * tf.math.exp(jumps[:, :, t]) - x[:, :, t ], axis=1, keepdims=True)
                #f_compensator = self.subnet[0].call(x[:, :, t],training=True) 
                ###
            else:
                z = self.subnetControl[t](x[:, :, t], training=True) #* self.bsde.getFsdeDiffusion(t,x[:, :, t])/ self.bsde.dim

                input_concat =  tf.concat([x[:,:,t], x[:,:,t] * tf.math.exp(jumps[:,:,t])], axis=1)
                f_Jump = self.subnetCompensator[t].call(input_concat,training=True) * ( x[:, :, t  ] * tf.math.exp(jumps[:, :, t]) - x[:, :, t ] )
                f_compensator = self.subnet[t].call(x[:, :, t],training=True)
 
            compensatedJump = f_Jump - f_compensator * self.bsde.delta_t
            penalty += ( tf.reduce_mean( compensatedJump ) )**2 
            
            
            y = y - self.bsde.delta_t * (self.bsde.f_tf(time_stamp[t], x[:, :, t], y, z)) + compensatedJump + \
               tf.reduce_sum(z * self.bsde.getFsdeDiffusion(time_stamp[t],x[:, :, t]) * dw[:, :, t], 1, keepdims=True) 
               
    
        return y, penalty 
       
    
    @tf.function
    def predict_step(self, data):
        print('Predicting!')
        x, Poisson, jumps, dw = data[0] 
        
        time_stamp = np.arange(0, self.eqn_config.num_time_interval) * self.bsde.delta_t
        all_one_vec = tf.ones(shape=tf.stack([tf.shape(Poisson)[0], 1]), dtype=self.net_config.dtype)
        
        
        y = all_one_vec * self.y_init
        #y = all_one_vec * self.subnet[0].call(all_one_vec * self.bsde.x_init,False)
           
       
        history = tf.TensorArray(self.net_config.dtype,size=self.bsde.num_time_interval+1)     
        history = history.write(0,y)
        
        one_net = False
        for t in range(0, self.bsde.num_time_interval):

            if one_net:
                batch_size = tf.shape(x)[0]
                
                t_vector = tf.cast(tf.fill([batch_size, 1], self.bsde.delta_t * t), dtype=x.dtype)

                input_concat_z = tf.concat([t_vector, x[:,:,t]], axis=1)
                z = self.subnetControl[0](input_concat_z, training=False) # * self.bsde.getFsdeDiffusion(t,x[:, :, t])/ self.bsde.dim
                
                #input_concat_jump =  tf.concat([t_vector, x[:,:,t], x[:, :, t ] * tf.math.exp(jumps[:, :, t])], axis=1)
                input_concat_jump =  tf.concat([t_vector, x[:,:,t]], axis=1)
                f_Jump = self.subnetCompensator[0].call(input_concat_jump, training=False) * ( x[:, :, t  ] * tf.math.exp(jumps[:, :, t]) - x[:, :, t ] ) 
                
                input_concat_comp = tf.concat([t_vector, x[:,:,t]], axis=1)
                f_compensator = self.subnet[0].call(input_concat_comp, training=False)  
                
                ###
                #f_Jump = self.subnetCompensator[0].call(x[:,:,t],training=False) * tf.reduce_sum( x[:, :, t ] * tf.math.exp(jumps[:, :, t]) - x[:, :, t], axis=1, keepdims=True)
                #f_compensator = self.subnet[0].call(x[:, :, t],training=False)
                ###
            else:
                z = self.subnetControl[t](x[:, :, t], training=False) #* self.bsde.getFsdeDiffusion(t,x[:, :, t])/ self.bsde.dim
                input_concat =  tf.concat([x[:,:,t], x[:,:,t] * tf.math.exp(jumps[:,:,t])], axis=1)
                f_Jump = self.subnetCompensator[t].call(input_concat,training=True) * ( x[:, :, t  ] * tf.math.exp(jumps[:, :, t]) - x[:, :, t ] )
                f_compensator = self.subnet[t].call(x[:, :, t],training=False)

            compensatedJump = f_Jump - f_compensator * self.bsde.delta_t

            y = y - self.bsde.delta_t * (
                self.bsde.f_tf(time_stamp[t], x[:, :, t], y, z)) + compensatedJump + \
               tf.reduce_sum(z * self.bsde.getFsdeDiffusion(time_stamp[t],x[:, :, t]) * dw[:, :, t], 1, keepdims=True)  
               

            history = history.write(t+1,y)

        history = tf.transpose(history.stack(),perm=[1,2,0])
        # return Poisson, jumps, x, history, dw    
        return x, Poisson, jumps, dw, history 

    def simulate_path(self,num_sample):
        # return self.predict(num_sample)[3]    
        return self.predict(num_sample)[4] 


class FeedForwardSubNet(tf.keras.Model):
    def __init__(self, config,dim):
        super(FeedForwardSubNet, self).__init__()        
        num_hiddens = config.net_config.num_hiddens
        self.bn_layers = [
            tf.keras.layers.BatchNormalization(
                momentum=0.99,
                epsilon=1e-6,
                beta_initializer=tf.random_normal_initializer(0.0, stddev=0.1),
                gamma_initializer=tf.random_uniform_initializer(0., 0.5)
            )
            for _ in range(len(num_hiddens) + 2)]
        self.dense_layers = [tf.keras.layers.Dense(num_hiddens[i],
                                                   use_bias=True,
                                                   activation=None,)
                             for i in range(len(num_hiddens))]
        self.dense_layers.append(tf.keras.layers.Dense(dim, activation=None))

    def call(self, x, training):
        """structure: bn -> (dense -> bn -> relu) * len(num_hiddens) -> dense """
        x = self.bn_layers[0](x, training)
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            x = self.bn_layers[i+1](x, training)
            x = tf.nn.relu(x)
            #x = tf.nn.sigmoid(x)
            #x = tf.nn.tanh(x)
        x = self.dense_layers[-1](x)

        
        """
        Questa somma serve per ridurre la dimensione da d a 1 ora la rete
        rappresenta una funzione da Rd a R.
        """
        #tf.math.reduce_sum(x,0)
        return x

    def grad(self, x):
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float64)
        with tf.GradientTape(watch_accessed_variables=True) as t:
            t.watch(x_tensor)
            out = self.call(x_tensor,training=False)
        grad = t.gradient(out,x_tensor)
        del t
        return grad

class FeedForwardSubNetControl(tf.keras.Model):
    def __init__(self, config,dim):
        super(FeedForwardSubNetControl, self).__init__()        
        num_hiddens = config.net_config.num_hiddens
        self.bn_layers = [
            tf.keras.layers.BatchNormalization(
                momentum=0.99,
                epsilon=1e-6,
                beta_initializer=tf.random_normal_initializer(0.0, stddev=0.1),
                gamma_initializer=tf.random_uniform_initializer(0., 0.5)
            )
            for _ in range(len(num_hiddens) + 2)]
        self.dense_layers = [tf.keras.layers.Dense(num_hiddens[i],
                                                   use_bias=True,
                                                   activation=None,)
                             for i in range(len(num_hiddens))]
        self.dense_layers.append(tf.keras.layers.Dense(dim, activation=None))

    def call(self, x, training):
        """structure: bn -> (dense -> bn -> relu) * len(num_hiddens) -> dense """
        x = self.bn_layers[0](x, training)
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            x = self.bn_layers[i+1](x, training)
            x = tf.nn.relu(x)
            #x = tf.nn.sigmoid(x)
            #x = tf.nn.tanh(x)
        x = self.dense_layers[-1](x)

        return x

    def grad(self, x):
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float64)
        with tf.GradientTape(watch_accessed_variables=True) as t:
            t.watch(x_tensor)
            out = self.call(x_tensor,training=False)
        grad = t.gradient(out,x_tensor)
        del t
        return grad
 
