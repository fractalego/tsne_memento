TSNEMemento: A wrapper for Laurens van der Maaten's parametric-tsne
===================================================================

T-SNE is a relevant techique for reducing the dimensionality of a
dataset. The original techique builds a low-dimension representation
trained upon a fixed set of vectors.

Parametric t-SNE allows the user to build a low-dimension
representation of the dataset that can be later accessed upon using
different vectors.

Parametric t-SNE has been developed entirely by [Laurens van der
Maaten](https://lvdmaaten.github.io/tsne/). The code presented here is
merely a python wrapper of the technique, based on [this
repo](https://github.com/kylemcdonald/Parametric-t-SNE).


Usage
-----


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


Requirements
------------

* numpy==1.11.2
* Theano==0.7.0


References
----------

* [Laurens van der Maaten's page](https://lvdmaaten.github.io/tsne/)
* [Parametric t-SNE in Theano](https://github.com/kylemcdonald/Parametric-t-SNE)