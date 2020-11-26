# utils.py
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from celluloid import Camera
import time

def generate_video(data, y, title, res=100):
    """
    data -> list of size L
        each element in the list is N x 2
        where N -> data point number
        2 -> values on the two dims

    y -> label for each of the datapoints
    """
    start = time.time()
    numpoints = len(data)
    camera = Camera(plt.figure(dpi=res))
    plt.title(title)
    for i in range(numpoints):
        plt.scatter(data[i][:,0], data[i][:,1], marker='x', alpha=0.3,c=y.numpy(), s=100)
        camera.snap()
    anim = camera.animate()
    anim.save(title+'.mp4')
    print('Saved to {}.mp4 in {} seconds'.format(title, time.time()-start))

from sklearn import svm
def separability(data, y):
    X = np.array([data[-1][:,0], data[-1][:,1]]).T
    clf = svm.SVC(kernel='linear')
    clf.fit(X, y)
    print(np.mean(clf.predict(X)==y))

def convolve(points, kernelSize=5):
    array = np.convolve(points, np.ones(kernelSize)/kernelSize, 'valid')
    return array

def cosine_pairwise(x):
    x = x.unsqueeze(0)
    x = x.permute((1, 2, 0))
    cos_sim_pairwise = F.cosine_similarity(x, x.unsqueeze(1), dim=-2)
    cos_sim_pairwise = cos_sim_pairwise.permute((2, 0, 1))
    return cos_sim_pairwise

"""
Code to generate using anim

import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np

fig, ax = plt.subplots()  
ax.grid()  
ax.axis([-1,1,-1,1])
sc = ax.scatter(data[0][:,0], data[0][:,1], c=y.numpy(), marker="x", alpha=0.3) # set linestyle to none

def plot(a, data):
    sc.set_offsets(data[a])
    sc.set_array(y.numpy())

ani = matplotlib.animation.FuncAnimation(fig, plot, fargs=(data,),
            frames=280, interval=1, repeat=True) 
writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)
ani.save('test.mp4', writer=writer)
plt.show()
"""