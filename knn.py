from scipy.spatial import distance
from scipy.stats import mode
import numpy as np

class KNNClassifier():
    
    """  
    A K-Nearest Neighbors classifier that classifies an array of ints into ints categories 
    """

    def predict(self, classified, unclassified, label_column, k = 5):

        """
        Receives a matrix of classified floats, a matrix of unclassified floats, an array of labels
        corresponding to the classification of the classified set, and optionaly the number of nearest neighbors to consider,
        then, the algorithm classifies each of the unclassified itens based on the most common label among its neighbors
        """

        labels = []
        for i in unclassified:
            distances = []
            for j in classified:
                j_index = np.where(classified == j)
                d = distance.euclidean(i, j)
                distances.append((d, label_column[j_index[0][0]]))
            distances = np.array(distances, dtype = [('distance', float), ('label', int)])
            distances = np.sort(distances, order = 'distance')
            mode(distances[:k]['label'])
            labels.append(mode(distances[:k]['label'])[0][0])
        return labels


if __name__ == "__main__":
    x = np.random.randint(0, 20, size = [30,3])
    y = np.random.randint(0, 4, size = 30)
    z = np.random.randint(0, 20, size = [30,3])
    knn = KNNClassifier()
    print(knn.predict(x, z, y))
