import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import argparse
import copy
from numpy.core.fromnumeric import repeat
from knn import KNN

has_converged = False

def KMeans(data, k):
    """
    Takes in unlabeled data and a number of means to compute.
    Step 1: Choose k data points at random to create a mean at.
    Step 2: For each data point, compute which mean is closest and assign it to that mean.
    Step 3: Recompute the mean based on the data points assigned to it.
    Repeat step 2 and 3 until convergence.
    Yields frames of the animation
    """
    random_indices = np.random.choice(len(data), size=k, replace=False)
    previous_means = []
    means = data[random_indices, :]

    def converged(curr, prev):
        if len(curr) != len(prev):
            return False
        for a, b in zip(curr, prev):
            if not np.array_equal(np.array(a), np.array(b)):
                return False
        global has_converged
        has_converged = True
        return True

    while not converged(means, previous_means):
        previous_means = copy.deepcopy(means)
        points_segregated = [[] for _ in range(k)]
        for point in data:
            min_dist = np.inf
            for mean_idx, mean in enumerate(means):
                dist = np.linalg.norm(mean - point)
                if dist < min_dist:
                    min_dist = dist
                    assigned_mean = mean_idx
            points_segregated[assigned_mean].append(point)
        yield means, points_segregated

        for i, points in enumerate(points_segregated):
            means[i] = sum(points) / len(points)
        yield means, points_segregated

parser = argparse.ArgumentParser()
parser.add_argument("-k", type=int, help="The K in Kmeans (the number of groups)", default=3)
args = parser.parse_args()

# I can run any implemented algorithm from here
# Let's say the total dataset comes from 3 normally distributed sources
data_source_0 = np.random.multivariate_normal((-0.5, -0.5), np.identity(2)*0.2, 100)
data_source_1 = np.random.multivariate_normal((1.5, 1.), np.identity(2)*0.2, 100)
data_source_2 = np.random.multivariate_normal((0.5, 1.5), np.identity(2)*0.2, 100)
unlabeled_data = np.append(data_source_0, data_source_1, 0)
unlabeled_data = np.append(unlabeled_data, data_source_2, 0)

# Just data fig
fig, ax = plt.subplots()
color_array = ["red", "green", "blue", "cyan", "magenta", "yellow", "black", "grey"]
data_scats = [ax.scatter([], [], s=10) for i in range(args.k)]
mean_scat = ax.scatter([], [], marker="*", s=250)

def init():
    ax.set_xlim(min(unlabeled_data[:, 0]) - 0.05, max(unlabeled_data[:, 0]) + 0.05)
    ax.set_ylim(min(unlabeled_data[:, 1]) - 0.05, max(unlabeled_data[:, 1]) + 0.05)
    return mean_scat, *data_scats
    
def animate(frame, generator):
    try:
        means, point_sets = next(generator)
    except StopIteration:
        return mean_scat, *data_scats
        
    mean_scat.set_offsets(means)
    mean_scat.set(facecolor=color_array, edgecolor="black")
    for i, (data_scat, point_set) in enumerate(zip(data_scats, point_sets)):
        data_scat.set_offsets(point_set)
        data_scat.set(color=color_array[i])
    return mean_scat, *data_scats

def frame_generator():
    i = 0
    global has_converged
    while not has_converged:
        yield i
        i += 1

generator = KMeans(unlabeled_data, k=args.k)
anim = FuncAnimation(fig, animate, frames=frame_generator, interval=100, fargs=(generator,), init_func=init, blit=True, repeat=True)
writergif = matplotlib.animation.PillowWriter(fps=10)

anim.save(f"animations/kmeans_anim_k{args.k}.gif", writer=writergif)
