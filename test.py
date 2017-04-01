import numpy as np

from tsne_memento import TSNEMemento


if __name__ == '__main__':

    # Generate random vectors
    vectors = np.array([np.random.uniform(low=-1, high=1, size=(50,))
                        for _ in range(300)])
    
    # Create the tsne representation
    tsne = TSNEMemento(vectors, perplexity=25, dim=2)

    # Access the tsne representation
    print(tsne[vectors[0]])

    # Save the trained tsne representation
    tsne.save('./vectors.tsne')

    # Load the trained tsne representation
    tsne_saved = TSNEMemento.load('./vectors.tsne')

    # Access the tsne representation
    print(tsne_saved[vectors[0]])
