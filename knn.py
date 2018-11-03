from scipy.spatial import distance
from scipy.stats import mode
import numpy as np

class KNNClassifier():

    def predict(self, classified, unclassified, label_column, k = 5):
        labels = []
        for i in unclassified:
            distances = []
            for j in classified:
                j_index = np.where(classified == j)
                d = distance.euclidean(i, j)
                distances.append((d, label_column[j_index[0][0]]))
            distances = np.array(distances, dtype = [('distance', float), ('label', int)])
            #print(distances)
            distances = np.sort(distances, order = 'distance')
            #print(distances)
            #print(distances[:k])
            mode(distances[:k]['label'])
            labels.append(mode(distances[:k]['label'])[0][0])
        return labels


if __name__ == "__main__":
    x = np.random.randint(0, 20, size = [30,3])
    y = np.random.randint(0, 4, size = 30)
    print(y)
    z = np.random.randint(0, 20, size = [30,3])
    print(x, z)
    knn = KNNClassifier()
    print(knn.predict(x, z, y))
