from tsne_memento.parametric_tsne import ParametricTsne


class TSNEMemento:
    
    def __init__(self, vectors, dim=5, perplexity=25):
        self.ptsne = ParametricTsne(vectors, perplexity=perplexity, dimension=dim)
        self.projected = self.ptsne.predict(vectors)
        
    def __getitem__(self, vector):
        return self.ptsne.predict([vector])[0]    
    
    def save(self, filename):
        import pickle    
        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

        
    @staticmethod
    def load(filename):
        import pickle
        with open(filename, 'rb') as input_file:
            return pickle.load(input_file)
