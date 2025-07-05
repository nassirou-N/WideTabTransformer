import argparse

def parameter_parser():
    # Experiment parameters
    parser = argparse.ArgumentParser(description='Smart Contract Vulnerability Detection Using Wide and TabTransformer Neural Network')

    parser.add_argument('filename', type=str, help="name of file to process")
    parser.add_argument('-vt', type=str, choices=['ts', 're', 'io'])
        
    # Optimiseur et learning rate
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate (reduced for TabTransformer)')
    parser.add_argument('-d', '--dropout', type=float, default=0.15, help='dropout rate (optimized for TabTransformer)')
    
    # Dimensions vectorielles
    parser.add_argument('--vec_length', type=int, default=200, help='vector dimension (increased for TabTransformer)')
    
    # Paramètres d'entraînement
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs (increased for TabTransformer)')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size (increased for TabTransformer)')
    
    # Paramètres spécifiques au TabTransformer
    parser.add_argument('--num_heads', type=int, default=8, help='number of attention heads in TabTransformer')
    parser.add_argument('--key_dim', type=int, default=64, help='dimension of keys/values in TabTransformer')
    parser.add_argument('--ff_dim', type=int, default=256, help='feed-forward dimension in TabTransformer')
    parser.add_argument('--num_layers', type=int, default=3, help='number of transformer layers')
    
    # Paramètres pour la division Wide/TabTransformer
    parser.add_argument('--wide_features', type=int, default=30, help='number of features for wide component')

    return parser.parse_args()