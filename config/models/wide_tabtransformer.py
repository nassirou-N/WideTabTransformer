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

    def augment_data(self, vectors, labels):
        """
        Augmentation de données équilibrée
        """
        positive_indices = np.where(labels == 1)[0]
        negative_indices = np.where(labels == 0)[0]
        
        print(f"Original dataset: {len(positive_indices)} positive, {len(negative_indices)} negative")
        
        pos_count = len(positive_indices)
        neg_count = len(negative_indices)
        
        augmented_vectors = []
        augmented_labels = []
        
        # Keep original data
        augmented_vectors.extend(vectors)
        augmented_labels.extend(labels)
        
        # Augmentation modérée
        if pos_count < neg_count:
            minority_indices = positive_indices
            minority_label = 1
            samples_needed = min(neg_count - pos_count, pos_count * 3 // 4)  # 75% du minority
        else:
            minority_indices = negative_indices
            minority_label = 0
            samples_needed = min(pos_count - neg_count, neg_count * 3 // 4)  # 75% du minority
        
        # Generate augmented samples
        for _ in range(samples_needed):
            idx = np.random.choice(minority_indices)
            original_vector = vectors[idx]
            augmented_vector = self.apply_augmentation_moderate(original_vector)
            augmented_vectors.append(augmented_vector)
            augmented_labels.append(minority_label)
        
        augmented_vectors = np.array(augmented_vectors)
        augmented_labels = np.array(augmented_labels)
        
        print(f"Augmented dataset: {len(augmented_vectors)} total samples")
        print(f"Positive samples: {np.sum(augmented_labels == 1)}")
        print(f"Negative samples: {np.sum(augmented_labels == 0)}")
        
        return augmented_vectors, augmented_labels

    def apply_augmentation(self, vector):
        """
        Augmentation plus modérée
        """
        augmented = vector.copy()
        """ 
        # Technique 1: Bruit plus léger
        noise_factor = 0.005  # Réduire de 0.01 à 0.005
        noise = np.random.normal(0, noise_factor, size=augmented.shape)
        augmented += noise
        
        # Technique 2: Scaling plus conservateur
        scale_factor = np.random.uniform(0.98, 1.02)  # Réduire de 0.95-1.05 à 0.98-1.02
        augmented += scale_factor
        
        # Technique 3: Dropout plus léger
        dropout_rate = 0.1  # Réduire de 0.05 à 0.02
        dropout_mask = np.random.random(augmented.shape) > dropout_rate
        augmented *= dropout_mask """
        # augmented_vector = vector.copy()
    
        # Add Gaussian noise with moderate intensity
        noise_std = 0.05  # 5% noise
        noise = np.random.normal(0, noise_std, size=vector.shape)
        augmented_vector = augmented_vector + noise
        return augmented_vector

    def build_model(self, inputs):
        wide = Normalization()(inputs[0])
        tab = Normalization()(inputs[1])

        # TabTransformer avec réduction modérée
        tab_transformer = TabTransformer(
            num_heads=10,     # Réduire de 12 à 10 (pas 8)
            key_dim=96,       # Réduire de 128 à 96 (pas 64)
            ff_dim=384,       # Réduire de 512 à 384 (pas 256)
            num_layers=5,     # Réduire de 6 à 5 (pas 4)
            dropout_rate=0.2  # Augmenter de 0.1 à 0.2 (pas 0.3)
        )(tab)

        # Flatten the TabTransformer output
        tab_transformer_flat = Flatten()(tab_transformer)

        # Architecture TabTransformer - Réduction progressive
        tab_reduced = Dense(200, activation='gelu',  # Réduire de 256 à 200
                        kernel_regularizer=l1_l2(l1=0.0008, l2=0.0008),  # Augmenter modérément
                        kernel_initializer='he_normal')(tab_transformer_flat)
        tab_reduced = BatchNormalization()(tab_reduced)
        tab_reduced = Dropout(0.25)(tab_reduced)  # Augmenter de 0.15 à 0.25

        tab_reduced = Dense(100, activation='gelu',  # Réduire de 128 à 100
                        kernel_regularizer=l1_l2(l1=0.0008, l2=0.0008))(tab_reduced)
        tab_reduced = BatchNormalization()(tab_reduced)
        tab_reduced = Dropout(0.2)(tab_reduced)  # Augmenter de 0.1 à 0.2

        tab_reduced = Dense(50, activation='gelu',  # Réduire de 64 à 50
                        kernel_regularizer=l1_l2(l1=0.0008, l2=0.0008))(tab_reduced)
        tab_reduced = BatchNormalization()(tab_reduced)
        tab_reduced = Dropout(0.15)(tab_reduced)  # Augmenter de 0.1 à 0.15

        # Flatten the wide input
        wide_flat = Flatten()(wide)

        # Partie Wide - Réduction modérée
        wide_processed = Dense(48, activation='gelu',  # Réduire de 64 à 48
                            kernel_regularizer=l1_l2(l1=0.0008, l2=0.0008))(wide_flat)
        wide_processed = BatchNormalization()(wide_processed)
        wide_processed = Dropout(0.2)(wide_processed)  # Augmenter de 0.1 à 0.2

        wide_processed = Dense(24, activation='gelu',  # Réduire de 32 à 24
                            kernel_regularizer=l1_l2(l1=0.0008, l2=0.0008))(wide_processed)
        wide_processed = BatchNormalization()(wide_processed)
        wide_processed = Dropout(0.15)(wide_processed)  # Augmenter de 0.1 à 0.15

        # Concatenate
        merged = Concatenate(axis=-1)([wide_processed, tab_reduced])
        
        # Couches finales - Réduction progressive
        final_dense = Dense(96, activation='gelu',  # Réduire de 128 à 96
                        kernel_regularizer=l1_l2(l1=0.0012, l2=0.0012))(merged)  # Augmenter modérément
        final_dense = BatchNormalization()(final_dense)
        final_dense = Dropout(0.3)(final_dense)  # Augmenter de 0.2 à 0.3

        final_dense = Dense(48, activation='gelu',  # Réduire de 64 à 48
                        kernel_regularizer=l1_l2(l1=0.0012, l2=0.0012))(final_dense)
        final_dense = BatchNormalization()(final_dense)
        final_dense = Dropout(0.25)(final_dense)  # Augmenter de 0.15 à 0.25

        final_dense = Dense(24, activation='gelu',  # Réduire de 32 à 24
                        kernel_regularizer=l1_l2(l1=0.0012, l2=0.0012))(final_dense)
        final_dense = BatchNormalization()(final_dense)
        final_dense = Dropout(0.2)(final_dense)  # Augmenter de 0.1 à 0.2
        
        output = Dense(2, activation='softmax')(final_dense)

        model = Model(inputs=inputs, outputs=output)
        
        # Optimiseur avec ajustements modérés
        optimizer = Adam(
            learning_rate=8e-5,  # Réduire de 1e-4 à 8e-5 (pas 5e-5)
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            clipnorm=0.8  # Réduire de 1.0 à 0.8 (pas 0.5)
        )
        
        model.compile(
            optimizer=optimizer, 
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )

        return model

    def train(self):
       # Callbacks équilibrés
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=20,  # Réduire de 25 à 20 (pas 15)
            restore_best_weights=True,
            verbose=1,
            mode='min'
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.25,  # Réduire modérément
            patience=7,   # Réduire de 10 à 7 (pas 5)
            min_lr=1e-7,
            verbose=1,
            mode='min'
        )

        checkpoint = ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )
        
        history = self.model.fit(
            ([self.x_train_wide, self.x_train_tab]), self.y_train,
            epochs=self.epochs,
            class_weight=self.class_weight,
            verbose=1,
            batch_size=self.batch_size,
            validation_split=0.27,  # Augmenter de 0.25 à 0.27 (pas 0.3)
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