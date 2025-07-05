import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from config.fragment_vectorizer import FragmentVectorizer
from config.models.wide_tabtransformer import WIDE_TabTransformer
import pandas as pd
from config.arg_parser import parameter_parser
import warnings
import numpy as np
import matplotlib.pyplot as plt

# Set print options to display the entire array
np.set_printoptions(threshold=np.inf)

warnings.filterwarnings("ignore")

args = parameter_parser()

for arg in vars(args):
    print(arg, getattr(args, arg))

def parse_file(filename):
    print('parsing file... (', filename, ')')
    with open(filename, "r", encoding="utf8") as file:
        fragment = []
        fragment_val = 0
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            if "-" * 40 in line and fragment:
                yield fragment, fragment_val
                fragment = []
            elif stripped.split()[0].isdigit():
                if fragment:
                    if stripped.isdigit():
                        fragment_val = int(stripped)
                    else:
                        fragment.append(stripped)
            else:
                fragment.append(stripped)

def get_vectors_df(filename, vec_len):
    fragments = []
    count = 0
    vectorizer = FragmentVectorizer(vec_len)

    for fragment, val in parse_file(filename):
        count += 1
        print("Collecting fragments...", count, end="\r")
        vectorizer.add_fragment(fragment)

        row = {"fragment": fragment, "val": val}
        fragments.append(row)
    
    print('Found {} forward slices and {} backward slices'.format(
        vectorizer.forward_slices, vectorizer.backward_slices))

    print("Training Word2Vec model...", end="\r")
    vectorizer.train_model()
    print()
    
    vectors = []
    count = 0

    for fragment in fragments:
        count += 1
        print("Processing fragments...", count, end="\r")
        vector = vectorizer.vectorize(fragment["fragment"])
        row = {"vector": vector, "val": fragment["val"]}
        vectors.append(row)

    # Convert vectors to dataframe
    df = pd.DataFrame(vectors)
    return df

def main():
    filename = args.filename        # smart contract source file
    vtype = args.vt                 # vulnerability type (re, ts, io)

    base = os.path.splitext(os.path.basename(filename))[0]
    vector_filename = base + "_fragment_vectors_tabtransformer.pkl"
    dataset = "config/train_data/" + vector_filename
    print(f"Dataset path: {dataset}\n")
    
    vector_length = args.vec_length
    
    if os.path.exists(dataset):
        print("Loading existing vector dataframe...")
        df = pd.read_pickle(dataset)
    else:
        print('Generating vector dataframe...')
        df = get_vectors_df(filename, vector_length)
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(dataset), exist_ok=True)
        # Convert dataframe to pickle
        df.to_pickle(dataset)
        print(f"Dataframe saved to {dataset}")

    print(f"Dataset shape: {df.shape}")
    print(f"Vector shape: {df.iloc[0, 0].shape}")
    print(f"Labels distribution: {df.iloc[:, 1].value_counts()}")
    
    # Initialize and train the Wide + TabTransformer model
    model = WIDE_TabTransformer(df, args)
    
    print("\nModel architecture:")
    model.model.summary()
    
    print("\nStarting training...")
    history = model.train()
    model.plot_training_curves(history)
    
    print("\nEvaluating model...")
    model.test()


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    main()