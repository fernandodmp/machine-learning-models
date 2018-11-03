from scipy.spatial import distance
from scipy.stats import mode
import numpy as np

class KNNClassifier():

    """  
    A K-Nearest Neighbors classifier that classifies an array of floats into ints categories 
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


class KNNRegressor():
    
    """  
    A K-Nearest Neighbors regressor that regresses an array of floats into a float value
    """

    def predict(self, regressed, unregressed, regression, k = 5):

        """
        Receives a matrix of regressed floats, a matrix of unregressed floats, an array of floats
        corresponding to the regression of the regressed set, and optionaly the number of nearest neighbors to consider,
        then, the algorithm regresses each of the unregressed itens based on the mean of its neighbors regressed value
        """ 

        values = []
        for i in unregressed:
            distances = []
            for j in regressed:
                j_index = np.where(regressed == j)
                d = distance.euclidean(i, j)
                distances.append((d, regression[j_index[0][0]]))
            distances = np.array(distances, dtype = [('distance', float), ('value', int)])
            distances = np.sort(distances, order = 'distance')
            mode(distances[:k]['value'])
            values.append(np.mean(distances[:k]['value']))
        return values

if __name__ == "__main__":
    x = np.random.randint(0, 20, size = [30,3])
    y = np.random.randint(0, 4, size = 30)
    z = np.random.randint(0, 20, size = [30,3])
    knn = KNNClassifier()
    print(knn.predict(x, z, y))

    ts_input = np.array([[1,2,3,4],
                         [1,5,7,8],
                         [1,10,5,3],
                         [10,1,1,3],
                         [1,10,5,1],
                         [7,5,6,2],
                         [0,0,0,0],
                         [3,5,7,0],
                         [5,5,5,5]])

    ts_output = np.array([[10,21,19,15,17,20,0, 15, 20]]).T 

    testing_data = np.array([[1,4,5,6],
                             [5,7,1,2],
                             [1,3,4,5],])

    knn = KNNRegressor()
    print(knn.predict(ts_input, testing_data, ts_output))
