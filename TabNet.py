import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow_addons.activations import sparsemax

def glu(x, feature_dim):
    return x[:,:feature_dim] * tf.nn.sigmoid(x[:, feature_dim:])

class FCBNBlock(keras.Model):
    def __init__(self, 
                 feature_dim,
                 block_name, 
                 bn_momentum):
        super().__init__()
        self.layer = layers.Dense(feature_dim,name=block_name)
        self.bn = layers.BatchNormalization(momentum=bn_momentum)

    def call(self, x, training=None):
        return self.bn(self.layer(x), training=training)

class TabNet(tf.keras.Model):

    def __init__(self,feature_columns,
    feature_dim,output_dim,num_features,
    num_decision_steps,
    relaxation_factor,sparsity_coefficient,
    batch_momentum,epsilon=1e-5):
        super().__init__()
        self.feature_columns = feature_columns
        self.num_features = num_features
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.num_decision_steps = num_decision_steps
        self.relaxation_factor = relaxation_factor
        self.sparsity_coefficient = sparsity_coefficient
        self.batch_momentum = batch_momentum
        self.epsilon = epsilon
        self.input_features = tf.keras.layers.DenseFeatures(feature_columns, trainable=True)
        self.input_bn = tf.keras.layers.BatchNormalization(axis=-1, momentum=batch_momentum, name='input_bn')
        self.transform_f1 = FCBNBlock(2*self.feature_dim,'shared_1', self.batch_momentum)
        self.transform_f2 = FCBNBlock(2*self.feature_dim,'shared_2', self.batch_momentum)
        self.transform_f3_list = [FCBNBlock(2*self.feature_dim, f'decision_1_{i}', self.batch_momentum) for i in range(self.num_decision_steps)]
        self.transform_f4_list = [FCBNBlock(2*self.feature_dim, f'decision_2_{i}', self.batch_momentum) for i in range(self.num_decision_steps)]
        self.transform_coef_list = [FCBNBlock(self.num_features, f'attention_{i}', self.batch_momentum) for i in range(self.num_decision_steps - 1)]

    def call(self, inputs, training=None):
        features = self.input_features(inputs)
        features = self.input_bn(features, training=training)
        batch_size = tf.shape(features)[0]

        output_aggregated = tf.zeros([batch_size, self.output_dim])
        masked_features = features
        mask_values = tf.zeros([batch_size, self.num_features])
        aggregated_mask_values = tf.zeros([batch_size, self.num_features])
        complementary_aggregated_mask_values = tf.ones([batch_size, self.num_features])

        total_entropy = 0.0
        entropy_loss = 0.0

        for ni in range(self.num_decision_steps):
            transform_f1 = self.transform_f1(masked_features, training=training)
            transform_f1 = glu(transform_f1, self.feature_dim)

            transform_f2 = self.transform_f2(transform_f1, training=training)
            transform_f2 = (glu(transform_f2, self.feature_dim) +
                            transform_f1) * tf.math.sqrt(0.5)

            transform_f3 = self.transform_f3_list[ni](transform_f2, training=training)
            transform_f3 = (glu(transform_f3, self.feature_dim) +
                            transform_f2) * tf.math.sqrt(0.5)

            transform_f4 = self.transform_f4_list[ni](transform_f3, training=training)
            transform_f4 = (glu(transform_f4, self.feature_dim) +
                            transform_f3) * tf.math.sqrt(0.5)

            if (ni > 0 or self.num_decision_steps == 1):
                decision_out = tf.nn.relu(transform_f4[:, :self.output_dim])

                output_aggregated += decision_out

                scale_agg = tf.reduce_sum(decision_out, axis=1, keepdims=True)

                if self.num_decision_steps > 1:
                  scale_agg = scale_agg / tf.cast(self.num_decision_steps - 1, tf.float32)
                aggregated_mask_values += mask_values * scale_agg

            features_for_coeff = transform_f4[:, self.output_dim:]

            if ni < (self.num_decision_steps - 1):

                mask_values = self.transform_coef_list[ni](features_for_coeff, training=training)
                mask_values *= complementary_aggregated_mask_values

                mask_values = sparsemax(mask_values, axis=-1)


                complementary_aggregated_mask_values *= (self.relaxation_factor - mask_values)

                total_entropy += tf.reduce_mean( tf.reduce_sum(-mask_values * tf.math.log(mask_values + self.epsilon), axis=1)) / (tf.cast(self.num_decision_steps - 1, tf.float32))

                entropy_loss = total_entropy

                masked_features = tf.multiply(mask_values, features)

            else:
                entropy_loss = 0.0

        self.add_loss(self.sparsity_coefficient * entropy_loss)


        return output_aggregated


class TabNetClassifier(tf.keras.Model):

    def __init__(self, feature_columns,feature_dim,output_dim,num_features,num_classes,num_decision_steps=5,relaxation_factor=1.5,sparsity_coefficient=1e-5,batch_momentum=0.98,epsilon=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.tabnet = TabNet(feature_columns=feature_columns,num_features=num_features,feature_dim=feature_dim,
                             output_dim=output_dim,num_decision_steps=num_decision_steps,relaxation_factor=relaxation_factor,
                             sparsity_coefficient=sparsity_coefficient,batch_momentum=batch_momentum,epsilon=epsilon)

        self.clf = tf.keras.layers.Dense(num_classes, activation='softmax', use_bias=False, name='classifier')

    def call(self, inputs, training=None):
        self.activations = self.tabnet(inputs, training=training)
        out = self.clf(self.activations)
        return out