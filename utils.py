import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
from skimage import filters, color
from scipy.ndimage.filters import laplace
from sys import getsizeof


def naive_average(images):
    k = len(images)
    output = None
    for im in images:
        if output is None:
            output = np.asarray(im).astype(float)
        else:
            output += np.asarray(im).astype(float)
    output /= k
    output = output.astype('uint8')
    return output


def load_images_from_path(pictures_dir, setup):
    img_paths = os.listdir(os.path.join(pictures_dir, setup))
    images = []
    for path in img_paths:
        pic_path = os.path.join(pictures_dir, setup, path)
        images.append(Image.open(pic_path))
    return images


def contrast(im):
    return np.abs(filters.laplace(filters.gaussian(im)))


def saturation(im):
    avg = np.mean(im, axis=(0, 1))
    saturation = abs(im - avg)
    # ici ils disent pas si il faut multiplier ou sommer, mais comme pour la well-exposedness
    # ils font une multiplication, je pense qu'il faut faire pareil ici
    return np.prod(saturation, axis=2)


def exposure(im, sigma=0.2):
    ans = im / 255.
    gaussian_array = np.array([0.5, 0.5, 0.5])
    ans = np.exp(- (0.5 / sigma ** 2) * ((ans - gaussian_array) ** 2))
    ans = np.prod(ans, axis=2)
    return ans


def weight_map(im, wc=1, ws=1, we=1):
    im = np.asarray(im).astype(float)
    im_gray = color.rgb2gray(im)

    ans = (exposure(im) ** we) * (saturation(im) ** ws) * (contrast(im_gray) ** wc)

    return ans


def normalize_weight_map(weight_maps):
    a = np.sum(weight_maps, axis=0)
    a = a.reshape((1,) + a.shape)
    a += 0.00000000001
    return weight_maps / a


def eq1(images, blur=False, sigma=10.):
    arrays = [np.asarray(im) for im in images]

    if blur:
        weight_maps = np.array([filters.gaussian(weight_map(a), sigma=sigma) for a in arrays])
    else:
        weight_maps = np.array([weight_map(a) for a in arrays])

    arrays = np.array(arrays)

    weight_maps = normalize_weight_map(weight_maps)

    output = np.reshape(weight_maps, weight_maps.shape + (1,)) * arrays
    output = np.sum(output, axis=0)

    return output.astype('uint8')


def normalize(im):
    return (im - np.min(im)) / (np.max(im) - np.min(im))


def gaussian_pyramid(im, sigma=15, n=10):
    ans = np.zeros((n,) + im.shape, dtype="f8")

    ans[0] = im
    for i in range(1, n):
        ans[i] = filters.gaussian(ans[i - 1], sigma=sigma)
    return ans


def gaussian_pyramid_yield(im, sigma=15, n=10):
    last_im = im.copy()
    yield last_im

    for i in range(1, n):
        last_im = filters.gaussian(last_im, sigma=sigma)
        yield last_im


def laplace_pyramid(gaussian_pyr):
    laplace_pyramid = np.zeros(gaussian_pyr.shape, dtype=gaussian_pyr.dtype)
    for i in range(gaussian_pyr.shape[0] - 1):
        laplace_pyramid[i] = gaussian_pyr[i] - gaussian_pyr[i + 1]

    laplace_pyramid[gaussian_pyr.shape[0] - 1] = gaussian_pyr[gaussian_pyr.shape[0] - 1]

    return laplace_pyramid


def laplace_pyramid_yield(im, sigma=15, n=10):

    current = None
    generator = gaussian_pyramid_yield(im, sigma=sigma, n=n)

    last_im = next(generator)

    for i in range(1, n):
        current = next(generator)

        yield last_im - current

        last_im = current

    yield current


def actual_size(x):
    return x / (8. * 1024)


def pyramid_merge(pictures_dir, setup, wc=1, we=1, ws=1, sigma=15, n=10):
    images = load_images_from_path(pictures_dir, setup)
    images = [np.asarray(im) for im in images]
    weight_maps = np.array([weight_map(a, wc=wc, we=we, ws=ws) for a in images])
    weight_maps = normalize_weight_map(weight_maps)

    weight_maps = np.array([gaussian_pyramid(weight_maps[i], n=n, sigma=sigma) for i in range(weight_maps.shape[0])])

    arr_gp = np.zeros((len(images), n) + images[0].shape, dtype="f8")
    for (i, a) in enumerate(images):
        arr_gp[i] = gaussian_pyramid(a, n=n, sigma=sigma)

    arr_lp = np.zeros(arr_gp.shape, dtype="f8")
    for (i, a) in enumerate(images):
        arr_lp[i] = laplace_pyramid(arr_gp[i])

    del arr_gp

    weight_maps = np.reshape(weight_maps, weight_maps.shape + (1,))
    product = weight_maps * arr_lp

    del weight_maps
    del arr_lp

    product = np.sum(product, axis=0)
    product = np.sum(product, axis=0)

    return product


def pyramid_merge_low_ram(pictures_dir, setup, wc=1, we=1, ws=1, sigma=15, n=10):
    images = load_images_from_path(pictures_dir, setup)
    images = [np.asarray(im).astype(float) for im in images]
    weight_maps = np.array([weight_map(a, wc=wc, we=we, ws=ws) for a in images])
    weight_maps = normalize_weight_map(weight_maps)

    product = np.zeros(images[0].shape).astype(float)

    for i in range(len(images)):
        print(i)
        gaussian_generator = gaussian_pyramid_yield(weight_maps[i], sigma=sigma, n=n)
        laplace_generator = laplace_pyramid_yield(images[i], sigma=sigma, n=n)

        for j in range(n):
            current_gaussian = next(gaussian_generator)
            current_gaussian = np.reshape(current_gaussian, current_gaussian.shape + (1,))
            current_laplace = next(laplace_generator)

            product += current_gaussian * current_laplace

    return product


if __name__ == "__main__":
    pass
