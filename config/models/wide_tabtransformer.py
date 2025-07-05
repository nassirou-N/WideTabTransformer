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

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
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

        # AUGMENTATION DE DONNÉES - CRITIQUE POUR 99%
        self.vectors, self.labels = self.augment_data(self.vectors, self.labels)

        positive_idxs = np.where(self.labels == 1)[0]
        negative_idxs = np.where(self.labels == 0)[0]

        idxs = np.concatenate([positive_idxs, negative_idxs])

        x_train, x_test, y_train, y_test = train_test_split(self.vectors[idxs], self.labels[idxs],
                                                            test_size=0.15, stratify=self.labels[idxs], random_state=42)
        split_point = 60  # AJUSTÉ
        # Split pour Wide (premières 30 features) et TabTransformer (dernières 70 features)
        self.x_train_wide, self.x_train_tab = x_train[:, :split_point], x_train[:, split_point:]
        self.x_test_wide, self.x_test_tab = x_test[:, :split_point], x_test[:, split_point:]

        self.y_train = to_categorical(y_train)
        self.y_test = to_categorical(y_test)

        classes = np.array([0, 1])
        class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=self.labels)
        class_weights[1] *= 1.5 
        self.class_weight = {index: weight for index, weight in enumerate(class_weights)}

        input_tab = Input(shape=(self.x_train_tab.shape[1], self.x_train_tab.shape[2]))
        input_wide = Input(shape=(self.x_train_wide.shape[1], self.x_train_wide.shape[2]))

        self.model = self.build_model(inputs=[input_wide, input_tab])

    def build_model(self, inputs):
        wide = Normalization()(inputs[0])
        tab = Normalization()(inputs[1])

        # Apply TabTransformer avec plus de dropout
        tab_transformer = TabTransformer(
            num_heads=12,
            key_dim=128,
            ff_dim=512,
            num_layers=6,
            dropout_rate=0.1  # Augmenté de 0.15 à 0.3
        )(tab)

         # Architecture plus profonde et complexe
        tab_reduced = Dense(256, activation='gelu',  # AUGMENTÉ et changé activation
                           kernel_regularizer=l1_l2(l1=0.0005, l2=0.0005),  # RÉDUIT
                           kernel_initializer='he_normal')(tab_transformer)
        tab_reduced = BatchNormalization()(tab_reduced)
        tab_reduced = Dropout(0.15)(tab_reduced)  # RÉDUIT

        tab_reduced = Dense(128, activation='gelu',
                           kernel_regularizer=l1_l2(l1=0.0005, l2=0.0005))(tab_reduced)
        tab_reduced = BatchNormalization()(tab_reduced)
        tab_reduced = Dropout(0.1)(tab_reduced)

        tab_reduced = Dense(64, activation='gelu',  # COUCHE AJOUTÉE
                           kernel_regularizer=l1_l2(l1=0.0005, l2=0.0005))(tab_reduced)
        tab_reduced = BatchNormalization()(tab_reduced)
        tab_reduced = Dropout(0.1)(tab_reduced)

        # Partie Wide plus complexe
        wide_processed = Dense(64, activation='gelu',  # AUGMENTÉ
                              kernel_regularizer=l1_l2(l1=0.0005, l2=0.0005))(wide)
        wide_processed = BatchNormalization()(wide_processed)
        wide_processed = Dropout(0.1)(wide_processed)

        wide_processed = Dense(32, activation='gelu',  # COUCHE AJOUTÉE
                              kernel_regularizer=l1_l2(l1=0.0005, l2=0.0005))(wide_processed)
        wide_processed = BatchNormalization()(wide_processed)
        wide_processed = Dropout(0.1)(wide_processed)

        # Flatten
        wide_flattened = Flatten()(wide_processed)
        tab_flattened = Flatten()(tab_reduced)
        
        # Fusion avec attention
        merged = Concatenate(axis=-1)([wide_flattened, tab_flattened])
        
        # Couches finales ultra-optimisées
        final_dense = Dense(128, activation='gelu',  # AUGMENTÉ
                           kernel_regularizer=l1_l2(l1=0.001, l2=0.001))(merged)
        final_dense = BatchNormalization()(final_dense)
        final_dense = Dropout(0.2)(final_dense)

        final_dense = Dense(64, activation='gelu',
                           kernel_regularizer=l1_l2(l1=0.001, l2=0.001))(final_dense)
        final_dense = BatchNormalization()(final_dense)
        final_dense = Dropout(0.15)(final_dense)

        final_dense = Dense(32, activation='gelu',  # COUCHE AJOUTÉE
                           kernel_regularizer=l1_l2(l1=0.001, l2=0.001))(final_dense)
        final_dense = BatchNormalization()(final_dense)
        final_dense = Dropout(0.1)(final_dense)
        
        output = Dense(2, activation='softmax')(final_dense)

        model = Model(inputs=inputs, outputs=output)
        
        # Optimiseur ultra-optimisé
        optimizer = Adam(
            learning_rate=1e-4,      # AJUSTÉ
            beta_1=0.9,              # AJUSTÉ
            beta_2=0.999,            # AJUSTÉ
            epsilon=1e-8,            # AJUSTÉ
            clipnorm=1.0             # AJOUTÉ - gradient clipping
        )
        
        model.compile(
            optimizer=optimizer, 
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']  # AJOUTÉ plus de métriques
        )

        return model

    def train(self):
        # Callbacks pour améliorer l'apprentissage
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=25,  # Arrêt si pas d'amélioration pendant 8 epochs
            restore_best_weights=True,
            verbose=1,
            mode ='max'
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,  # Divise le LR par 2
            patience=10,  # Réduit le LR si pas d'amélioration pendant 4 epochs
            min_lr=1e-7,
            verbose=1,
            mode = 'max'
        )

        checkpoint = ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        history = self.model.fit(
            ([self.x_train_wide, self.x_train_tab]), self.y_train,
            epochs=self.epochs,
            class_weight=self.class_weight,
            verbose=1,
            batch_size=self.batch_size,
            validation_split=0.25,  # Augmenté de 0.1 à 0.2
            callbacks=[early_stopping, reduce_lr, checkpoint],
            shuffle=True
        )
        
        return history
    
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
    
    
    def plot_training_curves(self, history):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Courbe d'accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)

        # Courbe de loss
        ax2.plot(history.history['loss'], label='Training Loss', color='blue')
        ax2.plot(history.history['val_loss'], label='Validation Loss', color='red')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

        # Analyse des résultats
        best_val_acc = max(history.history['val_accuracy'])
        best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1

        print(f"\nMeilleure validation accuracy: {best_val_acc:.4f} à l'epoch {best_epoch}")
        print(f"Training accuracy finale: {history.history['accuracy'][-1]:.4f}")
        print(f"Validation accuracy finale: {history.history['val_accuracy'][-1]:.4f}")
        print(f"Écart (overfitting): {history.history['accuracy'][-1] - history.history['val_accuracy'][-1]:.4f}")
