from __future__ import print_function
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
from config.arg_parser import parameter_parser
import tensorflow as tf
from tensorflow.keras.layers import Normalization, Concatenate, Reshape, Flatten, Dropout, Dense, Input, Layer
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical
from config.transformer.TabTransformer import TabTransformer
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.metrics import confusion_matrix
from sklearn.utils import compute_class_weight
from sklearn.model_selection import train_test_split
import numpy as np

# Set print options to display the entire array
np.set_printoptions(threshold=np.inf)
warnings.filterwarnings("ignore")

np.random.seed(42)
tf.random.set_seed(42)

args = parameter_parser()

class WIDE_TabTransformer:
    def __init__(self, data, args):
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.lr = args.lr

        self.vectors = np.stack(data.iloc[:, 0].values)
        self.labels = data.iloc[:, 1].values

        positive_idxs = np.where(self.labels == 1)[0]
        negative_idxs = np.where(self.labels == 0)[0]

        idxs = np.concatenate([positive_idxs, negative_idxs])

        x_train, x_test, y_train, y_test = train_test_split(self.vectors[idxs], self.labels[idxs],
                                                            test_size=0.2, stratify=self.labels[idxs], random_state=42)

        # Split pour Wide (premières 30 features) et TabTransformer (dernières 70 features)
        self.x_train_wide, self.x_train_tab = x_train[:, :30], x_train[:, 30:]
        self.x_test_wide, self.x_test_tab = x_test[:, :30], x_test[:, 30:]

        self.y_train = to_categorical(y_train)
        self.y_test = to_categorical(y_test)

        classes = np.array([0, 1])
        class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=self.labels)
        self.class_weight = {index: weight for index, weight in enumerate(class_weights)}

        input_tab = Input(shape=(self.x_train_tab.shape[1], self.x_train_tab.shape[2]))
        input_wide = Input(shape=(self.x_train_wide.shape[1], self.x_train_wide.shape[2]))

        self.model = self.build_model(inputs=[input_wide, input_tab])

    def build_model(self, inputs):
        wide = Normalization()(inputs[0])  # normalize the wide input tensor
        tab = Normalization()(inputs[1])   # normalize the tabular input tensor

        # Apply TabTransformer to the tabular component
        tab_transformer = TabTransformer(
            num_heads=8,        # Nombre de têtes d'attention
            key_dim=64,         # Dimension des clés/valeurs
            ff_dim=256,         # Dimension du feed-forward
            num_layers=3,       # Nombre de couches transformer
            dropout_rate=args.dropout
        )(tab)

        # Réduction dimensionnelle du TabTransformer
        tab_reduced = Dense(128, activation='relu')(tab_transformer)
        tab_reduced = Dropout(args.dropout)(tab_reduced)
        tab_reduced = Dense(64, activation='relu')(tab_reduced)

        # Traitement de la partie wide
        wide_processed = Dense(32, activation='relu')(wide)
        wide_processed = Dropout(args.dropout)(wide_processed)

        wide_flattened = Flatten()(wide_processed)
        tab_flattened = Flatten()(tab_reduced)
    
    # Fusion des composants wide et TabTransformer
        merged = Concatenate(axis=-1)([wide_flattened, tab_flattened])
        
        # Couches finales
        final_dense = Dense(128, activation='relu')(merged)
        final_dense = Dropout(args.dropout)(final_dense)
        final_dense = Dense(64, activation='relu')(final_dense)
        final_dense = Dropout(args.dropout)(final_dense)
        
        output = Dense(2, activation='softmax')(final_dense)

        model = Model(inputs=inputs, outputs=output)

        optimizer = Adam(learning_rate=self.lr)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def train(self):
        self.model.fit(
            ([self.x_train_wide, self.x_train_tab]), self.y_train,
            epochs=self.epochs, 
            class_weight=self.class_weight, 
            verbose=1, 
            batch_size=self.batch_size,
            validation_split=0.1
        )

    def test(self):
        values = self.model.evaluate([self.x_test_wide, self.x_test_tab], self.y_test, 
                                   batch_size=self.batch_size, verbose=0)
        print("\nAccuracy: ", values[1])
        predictions = (self.model.predict([self.x_test_wide, self.x_test_tab], 
                                        batch_size=self.batch_size, verbose=0)).round()

        tn, fp, fn, tp = confusion_matrix(np.argmax(self.y_test, axis=1), 
                                        np.argmax(predictions, axis=1)).ravel()
        print('False positive rate(FP): ', fp / (fp + tn))
        print('False negative rate(FN): ', fn / (fn + tp))
        recall = tp / (tp + fn)
        print('Recall: ', recall)
        precision = tp / (tp + fp)
        print('Precision: ', precision)
        print('F1 score: ', (2 * precision * recall) / (precision + recall))