import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization, MultiHeadAttention
from config.arg_parser import parameter_parser

args = parameter_parser()

class TabTransformer(Layer):
    def __init__(self, 
                 num_heads=10, 
                 key_dim=96, 
                 ff_dim=384, 
                 num_layers=5, 
                 dropout_rate=0.2, 
                 **kwargs):
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        super(TabTransformer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.feature_dim = input_shape[-1]
        
        # Projection layer to convert features to transformer dimension
        self.projection = Dense(self.key_dim * self.num_heads, activation='gelu', kernel_initializer='he_normal')
        
        # Transformer blocks
        self.transformer_blocks = []
        for i in range(self.num_layers):
            layer_dropout = self.dropout_rate * (1 + i * 0.15)
            self.transformer_blocks.append({
                'mha': MultiHeadAttention(
                    num_heads=self.num_heads,
                    key_dim=self.key_dim,
                    dropout=layer_dropout,
                    kernel_initializer='glorot_uniform'
                ),
                'ffn': [
                    Dense(self.ff_dim, activation='gelu', kernel_initializer='he_normal'),
                    Dropout(layer_dropout),
                    Dense(self.key_dim * self.num_heads, kernel_initializer='glorot_uniform')
                ],
                'layernorm1': LayerNormalization(epsilon=1e-6),
                'layernorm2': LayerNormalization(epsilon=1e-6),
                'dropout1': Dropout(layer_dropout),
                'dropout2': Dropout(layer_dropout)
            })
        
        # CORRIGÉ: Réduire la dimension de sortie finale
        self.final_projection = Dense(self.key_dim, activation='gelu', kernel_initializer='he_normal')
        
        super(TabTransformer, self).build(input_shape)
    
    def call(self, inputs, training=None, **kwargs):
        # Project input to transformer dimension
        x = self.projection(inputs)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            # Multi-head attention
            attention_output = block['mha'](x, x, training=training)
            attention_output = block['dropout1'](attention_output, training=training)
            x1 = block['layernorm1'](x + attention_output)
            
            # Feed forward network
            ffn_output = x1
            for layer in block['ffn']:
                ffn_output = layer(ffn_output, training=training) if hasattr(layer, 'training') else layer(ffn_output)
            
            ffn_output = block['dropout2'](ffn_output, training=training)
            x = block['layernorm2'](x1 + ffn_output)
        
        # Final projection
        output = self.final_projection(x)
        
        return output
    
    def compute_output_shape(self, input_shape):
        # CORRIGÉ: Mettre à jour la forme de sortie
        return input_shape[:-1] + (self.key_dim,)
    
    def get_config(self):
        config = super(TabTransformer, self).get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "ff_dim": self.ff_dim,
            "num_layers": self.num_layers,
            "dropout_rate": self.dropout_rate,
        })
        return config