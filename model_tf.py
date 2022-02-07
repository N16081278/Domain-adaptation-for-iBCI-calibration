import tensorflow as tf
import tensorflow.keras as keras
from loss_functions import maximumMeanDiscrepancy


class Extractor(keras.Model):
    def __init__(self):
        super().__init__()
        self.extractor = keras.layers.Bidirectional(keras.layers.GRU(256, dropout=0.2), merge_mode='concat')
        self.reduction = keras.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(512),
        ]) 

    def call(self, input_data):
        
        embedding = self.extractor(input_data)

        embedding = self.reduction(embedding)
       
        return embedding


class Decoder(keras.Model):
    def __init__(self):
        super().__init__()
        self.decoder_x = keras.Sequential([            
            keras.layers.Dense(64, activation=None),
            keras.layers.PReLU(),
            keras.layers.Dense(1)
        ])
        self.decoder_y = keras.Sequential([        
            keras.layers.Dense(64, activation=None),
            keras.layers.PReLU(),
            keras.layers.Dense(1)
        ])

    def call(self, embedding):
        
        output_x = self.decoder_x(embedding)
        output_y = self.decoder_y(embedding)

        output = tf.concat([output_x, output_y], axis=1)
        
        return output

class GradReverse(keras.layers.Layer):
    def __init__(self):
        super().__init__()

        self.alpha = 0.1

    @tf.custom_gradient
    def grad_reverse(self, x):
        y = tf.identity(x)
        def custom_grad(dy):
            return -dy * self.alpha
        return y, custom_grad

    def call(self, x):
        inputs = x       
        
        return self.grad_reverse(inputs)

class Discriminator(keras.Model):
    def __init__(self):
        super().__init__()
        self.discriminator = keras.Sequential([
            keras.layers.Dense(64, activation=None),
            keras.layers.PReLU(),
            keras.layers.Dense(2)
        ])
        
        self.reverLayer = GradReverse()

        self.alpha = 0.1

    def call(self, inputs):
        self.reverLayer.alpha = self.alpha
        x = self.reverLayer(inputs)
        x = self.discriminator(x)        
        return x

#######################################################################################
#######################################################################################

class simpleDecodeModel(keras.Model):
    def __init__(self):
        super().__init__()

        self.extractor = Extractor()
        self.decoder = Decoder()

        # params
        self.return_feature = False
    
    def call(self, inputs):
    
        feature = self.extractor(inputs)

        x = self.decoder(feature)

        if self.return_feature:
            return x, feature
        else:
            return x

#


class DeepDomainConfusionModel(keras.Model):
    def __init__(self, lambda_value = 0.25):
        super().__init__()

        self.extractor = Extractor()
        self.decoder = Decoder()

        self.loss_tracker_decode = keras.metrics.Mean(name='decode_loss')
        self.loss_tracker_domain = keras.metrics.Mean(name='domain_loss')

        # params
        self.predict_movement = False
        self.useTargetLabel = True
        self.lambda_value = lambda_value
    
    def train_step(self, data):
        x, y = data

        # split dataset
        source_x, target_x = x['source'], x['target']
        merge_x = tf.concat((source_x, target_x), axis=0)
        
        source_y_movement, target_y_movement = y['source_movement'], y['target_movement']       
        merge_y_movement = tf.concat((source_y_movement, target_y_movement), axis=0)

        with tf.GradientTape() as tape:
            # decode - Forward pass
            if self.useTargetLabel:
                feature = self.extractor(merge_x, training=True)
                y_pred = self.decoder(feature, training=True)
                loss_decode = keras.losses.mean_squared_error(merge_y_movement, y_pred)
            else:
                feature = self.extractor(source_x, training=True)
                y_pred = self.decoder(feature, training=True)
                loss_decode = keras.losses.mean_squared_error(source_y_movement, y_pred)
            
            # domain - Forward pass
            feature_source = self.extractor(source_x, training=True)
            feature_target = self.extractor(target_x, training=True)
            loss_domain = maximumMeanDiscrepancy(feature_source, feature_target)
            
            # combine loss
            loss = loss_decode + self.lambda_value * loss_domain
            

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.loss_tracker_decode.update_state(loss_decode)
        self.loss_tracker_domain.update_state(loss_domain)


        return {
            'decode_loss': self.loss_tracker_decode.result(), 
            'domain_loss': self.loss_tracker_domain.result(),            
        }

    

    def call(self, x):
        
        source_x, target_x = x['source'], x['target']       
        
        if self.predict_movement:
            feature_target = self.extractor(target_x)
            pred = self.decoder(feature_target)

            return pred
        else:
            feature_source = self.extractor(source_x)
            feature_target = self.extractor(target_x)

            return feature_source, feature_target

class DomainAdversarialNetwork(keras.Model):
    def __init__(self):
        super().__init__()

        self.encoder = Extractor()
        self.decoder = Decoder()

        self.discriminator = Discriminator()

        self.binaryCrossEntropyLoss = keras.losses.BinaryCrossentropy(from_logits=True)
        self.meanSquareLoss = keras.losses.MeanSquaredError()

        self.loss_tracker_decode = keras.metrics.Mean(name='decode_loss')
        self.loss_tracker_domain = keras.metrics.Mean(name='domain_loss')
        self.alpha = 0.1
        self.predict_movement = False
        self.useTargetLabel = True
    
    def train_step(self, data):
        x, y = data

        # split dataset
        source_x, target_x = x['source'], x['target']
        merge_x = tf.concat((source_x, target_x), axis=0)
        
        source_y_movement, target_y_movement = y['source_movement'], y['target_movement']
        source_y_domain, target_y_domain = y['source_domain'], y['target_domain']    
        merge_y_movement = tf.concat((source_y_movement, target_y_movement), axis=0)
        merge_y_domain = tf.concat((source_y_domain, target_y_domain), axis=0)

        # update params
        self.discriminator.alpha = self.alpha

        with tf.GradientTape() as tape:
            # decode - Forward pass
            if self.useTargetLabel:
                feature_decode = self.encoder(merge_x, training=True)
                y_pred_movement = self.decoder(feature_decode, training=True)  

                loss_decode = self.meanSquareLoss(merge_y_movement, y_pred_movement)
            else:
                feature_decode = self.encoder(source_x, training=True)
                y_pred_movement = self.decoder(feature_decode, training=True)  

                loss_decode = self.meanSquareLoss(source_y_movement, y_pred_movement)

            # domain - Forward pass
            feature_domain = self.encoder(merge_x, training=True)
            y_pred_domain = self.discriminator(feature_domain, training=True)
            loss_domain = self.binaryCrossEntropyLoss(merge_y_domain, y_pred_domain)
           
            # combine loss
            loss = loss_decode + loss_domain
            

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.loss_tracker_decode.update_state(loss_decode)
        self.loss_tracker_domain.update_state(loss_domain)

        return {
            'decode_loss': self.loss_tracker_decode.result(), 
            'domain_loss': self.loss_tracker_domain.result(),            
        }

    def call(self, x):
        source_x, target_x = x['source'], x['target']       
        
        if self.predict_movement:
            feature_target = self.encoder(target_x)
            pred = self.decoder(feature_target)
            
            return pred
        else:
            feature_source = self.encoder(source_x)
            feature_target = self.encoder(target_x)

            return feature_source, feature_target