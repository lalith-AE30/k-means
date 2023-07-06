from random import choices
from numba import jit
from numba import cuda
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

COLOR_LIMIT = 16777216

@jit(nopython=True)
def k_means_step(x: np.ndarray, k: np.ndarray, clusters: np.ndarray):
    for j, p in enumerate(x):
        dist = COLOR_LIMIT
        i_min = -1
        for i, m in enumerate(k):
            norm = np.linalg.norm(p-m)
            if norm < dist:
                dist = norm
                i_min = i
        clusters[i_min][j][3] = 1
        clusters[i_min][j][:3] = p
    k_new = np.copy(k)
    for i, _ in enumerate(k):
        cluster_size = np.sum(clusters[i, :, 3])
        if cluster_size:
            k_new[i] = (np.sum(clusters[i], axis=0)/cluster_size)[:3]
    return k_new, clusters


@cuda.jit
def kmeans_step_cuda_kernel(x: np.ndarray, k: np.ndarray, clusters: np.ndarray):
    pos = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    dist = COLOR_LIMIT
    i_min = -1
    for i, m in enumerate(k):
        norm = 0
        for j in range(3):
            norm += (x[pos][j]-m[j])**2
        if norm < dist:
            dist = norm
            i_min = i
    clusters[i_min][pos][3] = 1
    for j in range(3):
        clusters[i_min][pos][j] = x[pos][j]


def kmeans_step_cuda(x, k, clusters):
    threadsperblock = 256
    blockspergrid = (len(x) + (threadsperblock - 1)) // threadsperblock

    x_gpu = cuda.to_device(x)
    k_gpu = cuda.to_device(k)
    clusters = cuda.to_device(clusters)

    kmeans_step_cuda_kernel[blockspergrid,
                            threadsperblock](x_gpu, k_gpu, clusters)
    cuda.synchronize()

    clusters = clusters.copy_to_host()

    k_new = np.copy(k)
    for i, _ in enumerate(k):
        cluster_size = np.sum(clusters[i, :, 3])
        if cluster_size:
            k_new[i] = (np.sum(clusters[i], axis=0)/cluster_size)[:3]

    return k_new, clusters


def kmeans(x, k, cuda=False):
    x = np.array(x, dtype=np.float32)
    k = np.array(k, dtype=np.float32)
    clusters = np.zeros((len(k), len(x), 4), dtype=np.float32)

    if cuda:
        k_new, clusters = kmeans_step_cuda(x, k, clusters)
    else:
        k_new, clusters = k_means_step(x, k, clusters)

    while not np.array_equal(k, k_new):
        k = np.copy(k_new)
        if cuda:
            k_new, clusters = kmeans_step_cuda(x, k, clusters)
        else:
            k_new, clusters = k_means_step(x, k, clusters)
    return k_new, clusters


if __name__ == '__main__':

    # Helper Function    
    def rgb_to_hex(rgb):
        r, g, b = rgb
        return f"#{r:02x}{g:02x}{b:02x}"

    image = cv.imread(r'/path/to/image')

    fig = plt.figure()
    color_space = fig.add_subplot(121, projection='3d')
    preview = fig.add_subplot(122)
    preview.imshow(image[:, :, ::-1])

    SCALE = (100_000**0.5)/max(image.shape)
    image = cv.resize(image, (0, 0), fx=SCALE, fy=SCALE,
                    interpolation=cv.INTER_AREA)
    colors = np.reshape(
        image, (image.shape[0]*image.shape[1], image.shape[2]), order='C')

    MEANS = 16

    # Common Initialization methods 
    initial_k = [np.random.randint(0, 255, 3, dtype=np.uint8)
                for _ in range(MEANS)]
    # initial_k = choices(colors, k=MEANS)

    color_mean_array, clusters = kmeans(colors, initial_k, cuda=True)
    color_mean_array = np.array(color_mean_array, dtype=np.uint8)
    color_means = [rgb_to_hex(color_mean_array[i][::-1]) for i in range(MEANS)]

    print(color_means)
    color_space.set_xlabel('Blue')
    color_space.set_ylabel('Green')
    color_space.set_zlabel('Red')
    for i in range(MEANS):
        if sum(clusters[i, :, 3]):
            bgr = np.array(
                choices(clusters[i]
                        [(clusters[i, :, 3] == 1)], k=int(2000/MEANS
                                                        )))
            b, g, r = [bgr[:, c] for c in range(3)]
            color_space.scatter(b, g, r, c=color_means[i])

    color_space.legend(color_means, bbox_to_anchor=(0, 0), loc="lower left",
                    bbox_transform=plt.gcf().transFigure)
    plt.show()
