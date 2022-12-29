import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

x_train, x_test, y_train, y_test = np.load('classification_data.npy', allow_pickle=True)

# mean vectors

n0 = 0
n1 = 0
m0 = np.zeros((1, x_train.shape[1]))
m1 = np.zeros((1, x_train.shape[1]))

for i in range(0, x_train.shape[0]):
    if y_train[i] == 0:
        m0 = m0 + x_train[i]
        n0 += 1
    elif y_train[i] == 1:
        m1  = m1 + x_train[i]
        n1 += 1

m0 /= n0
m1 /= n1

print("m0: ")
print(m0)
print("m1: ")
print(m1)

# within class scatter matrix

s0 = np.zeros((x_train.shape[1], x_train.shape[1]))
s1 = np.zeros((x_train.shape[1], x_train.shape[1]))

for i in range(0, x_train.shape[0]):
    if y_train[i] == 0:
        s0 = s0 + (x_train[i] - m0).T @ (x_train[i] - m0)
    elif y_train[i] == 1:
        s1 = s1 + (x_train[i] - m1).T @ (x_train[i] - m1)

sw = s0 + s1

print("sw: ")
print(sw)

# between class scatter matrix

sb = (m1 - m0).T @ (m1 - m0)
print("sb: ")
print(sb)

# fisher linear discriminant

w = np.linalg.inv(sw) @ (m1 - m0).T
w /= np.linalg.norm(w)

print("w: ")
print(w)

# KNN

def knn(dataset, target, n): # return a list that contains the k-nearest-neighbors of the given target
    dist_list = [np.absolute(target - data) for data in dataset]

    enum_dist = enumerate(dist_list)
    sorted_neigh = sorted(enum_dist, key=lambda x: x[1])[:n]

    ind_list = []
    for tup in sorted_neigh:
        ind_list.append(tup[0])
    
    return np.array(ind_list)

x_train_proj = x_train @ w
x_test_proj = x_test @ w
for n in range(1, 6):
    y_pred = np.zeros(y_test.shape)
    for i in range(x_test_proj.size):
        target = x_test_proj[i]
        nc0 = 0 # neighbor of class 0
        nc1 = 0 # neighbor of class 1

        knn_list = knn(x_train_proj, target, n)
        for j in knn_list:
            if y_train[j] == 0:
                nc0 += 1
            elif y_train[j] == 1:
                nc1 += 1

        if (nc1 >= nc0):
            y_pred[i] = 1

    acc = accuracy_score(y_test, y_pred)
    print(f"k: {n}; acc: {acc}")

# plot 

w = w.reshape((2, ))
x_train_projected = x_train_proj * w

p1 = [x_train_projected.T[0][0], x_train_projected.T[1][0]]
p2 = [x_train_projected.T[0][1], x_train_projected.T[1][1]]

slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
intercept = p1[1] - slope * p1[0]

plt.title(f"Slope: {slope}, intercept: {intercept}", fontsize=9)

plt.plot(x_train_projected.T[0], x_train_projected.T[1], linewidth=0.7, zorder=5)

for i in range(x_train.shape[0]):
    x_values = [x_train.T[0][i], x_train_projected.T[0][i]]
    y_values = [x_train.T[1][i], x_train_projected.T[1][i]]
    plt.plot(x_values, y_values, linestyle="--", linewidth=0.1, c='k', zorder=0)

color = np.where(y_train == 1, 'r', 'b')
plt.scatter(x_train.T[0], x_train.T[1], c=color, s=1, zorder=10)
plt.scatter(x_train_projected.T[0], x_train_projected.T[1], c=color, s=1, zorder=10)

plt.show()