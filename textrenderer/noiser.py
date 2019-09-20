import numpy as np
import cv2


# https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
class NoiseAdd(object):
    """
    add noise
    """
    def __init__(self, cfg):
        self.cfg = cfg

    def apply(self, img):
        """
        :param img:  word image with big background
        """

        p = []
        funcs = []
        if self.cfg.noise.gauss.enable:
            p.append(self.cfg.noise.gauss.fraction)
            funcs.append(self.apply_gauss_noise)

        if self.cfg.noise.uniform.enable:
            p.append(self.cfg.noise.uniform.fraction)
            funcs.append(self.apply_uniform_noise)

        if self.cfg.noise.salt_pepper.enable:
            p.append(self.cfg.noise.salt_pepper.fraction)
            funcs.append(self.apply_sp_noise)

        if self.cfg.noise.poisson.enable:
            p.append(self.cfg.noise.poisson.fraction)
            funcs.append(self.apply_poisson_noise)

        if len(p) == 0:
            return img

        noise_func = np.random.choice(funcs, p=p)

        return noise_func(img)

    @staticmethod
    def apply_gauss_noise(img):
        """
        Gaussian-distributed additive noise.
        """
        shape = img.shape
        row, col = shape[:2]
        mean = 0
        stddev = np.sqrt(15)
        if len(shape) == 2:
            gauss_noise = np.zeros((row, col))
        else:
            gauss_noise = np.zeros((row, col,shape[2]))
        cv2.randn(gauss_noise, mean, stddev)
        out = img + gauss_noise

        return out

    @staticmethod
    def apply_uniform_noise(img):
        """
        Apply zero-mean uniform noise
        """
        shape = img.shape
        row, col = shape[:2]
        alpha = 0.05
        if len(shape) == 2:
            gauss = np.random.uniform(0 - alpha, alpha, (row, col))
            gauss = gauss.reshape(row, col)
        else:
            gauss = np.random.uniform(0 - alpha, alpha, (row, col, shape[2]))
            gauss = gauss.reshape(row, col, shape[2])
        out = img + img * gauss
        return out

    @staticmethod
    def apply_sp_noise(img):
        """
        Salt and pepper noise. Replaces random pixels with 0 or 255.
        """

        s_vs_p = 0.5
        amount = np.random.uniform(0.004, 0.01)
        out = np.copy(img)
        # Salt mode
        num_salt = np.ceil(amount * img.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in img.shape]
        out[tuple(coords)] = 255.

        # Pepper mode
        num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in img.shape]
        out[tuple(coords)] = 0
        # print("apply sp noise")
        return out

    @staticmethod
    def apply_poisson_noise(img):
        """
        Poisson-distributed noise generated from the data.
        """
        vals = len(np.unique(img))
        vals = 2 ** np.ceil(np.log2(vals))

        if vals < 0:
            return img

        noisy = np.random.poisson(img * vals) / float(vals)
        return noisy
