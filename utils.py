import itertools
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

def get_patch_dimensions(canvas_size, transform):
    N = canvas_size[0]
    x = [np.linspace(1, canvas_size[0], N, endpoint=True), np.ones(canvas_size[1]) * canvas_size[0],
         np.linspace(1, canvas_size[0], N, endpoint=True), np.ones(canvas_size[1])]
    x = np.array(list(itertools.chain.from_iterable(x))) - 1

    N = canvas_size[1]
    y = [np.ones(canvas_size[0]), np.linspace(1, canvas_size[1], N, endpoint=True),
         np.ones(canvas_size[0]) * canvas_size[1], np.linspace(1, canvas_size[1], N, endpoint=True)]
    y = np.array(list(itertools.chain.from_iterable(y))) - 1

    points = np.array([x, y]).transpose()
    transformed_points = transform_points(points, transform)

    width = int(np.max([np.max(transformed_points[:, 0]) + 1, canvas_size[1]]))
    height = int(np.max([np.max(transformed_points[:, 1]) + 1, canvas_size[0]]))
    return [height, width]


def transform_points(points, matrix):
	pts_pad = np.hstack([points, np.ones((points.shape[0], 1))])
	points_warp = np.dot(pts_pad, matrix.T)
	return points_warp[:, :-1]


def apply_transform(image, transform):
    inv_transform = inv(transform)
    new_image_shape = get_patch_dimensions([image.shape[1], image.shape[0]], inv_transform)
    new_image = np.zeros(new_image_shape)
    for iRow in range(image.shape[0]):
        for iCol in range(image.shape[1]):
            coord = np.array([[iCol, iRow]])
            transform_coord = transform_points(coord, inv_transform)
            if (not (transform_coord < 0).any()) and (transform_coord[0, 0] < image.shape[1]) and (transform_coord[0, 1] < image.shape[0]):
                new_image[iRow, iCol] = image[int(transform_coord[0, 1]), int(transform_coord[0, 0])]
                # new_image[int(transform_coord[0, 1]), int(transform_coord[0, 0])] = image[iRow, iCol]

    transformed_image = new_image[0:image.shape[0], 0:image.shape[1]]

    _, axs = plt.subplots(1, 2, figsize=(15, 10))
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title("Input Image")
    axs[1].imshow(transformed_image, cmap='gray')
    axs[1].set_title("Transformed Image")
    plt.show()

    return transformed_image
