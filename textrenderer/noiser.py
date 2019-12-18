import numpy as np
from PIL import Image
import noise
import cv2


# https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
class Noiser(object):
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

    def apply_gauss_noise(self, img):
        """
        Gaussian-distributed additive noise.
        """
        row, col, channel = img.shape

        mean = 0
        stddev = np.sqrt(15)
        gauss_noise = np.zeros((row, col, channel))
        cv2.randn(gauss_noise, mean, stddev)
        out = img + gauss_noise

        return out

    def apply_uniform_noise(self, img):
        """
        Apply zero-mean uniform noise
        """
        row, col, channel = img.shape
        alpha = 0.05
        gauss = np.random.uniform(0 - alpha, alpha, (row, col, channel))
        gauss = gauss.reshape(row, col, channel)
        out = img + img * gauss
        return out

    def apply_sp_noise(self, img):
        """
        Salt and pepper noise. Replaces random pixels with 0 or 255.
        """
        row, col, channel = img.shape
        s_vs_p = 0.5
        amount = np.random.uniform(0.004, 0.01)
        out = np.copy(img)
        # Salt mode
        num_salt = np.ceil(amount * img.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in img.shape]
        out[coords] = 255.

        # Pepper mode
        num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in img.shape]
        out[coords] = 0
        return out

    def apply_poisson_noise(self, img):
        """
        Poisson-distributed noise generated from the data.
        """
        vals = len(np.unique(img))
        vals = 2 ** np.ceil(np.log2(vals))

        if vals < 0:
            return img

        noisy = np.random.poisson(img * vals) / float(vals)
        return noisy


class Texture(object):

    def __init__(self, cfg):
        self.cfg = cfg

    def apply(self, img):
        """
        :param img:  text only img
        """

        p = []
        funcs = []
        if self.cfg.texture.cloud.enable:
            p.append(self.cfg.texture.cloud.fraction)
            funcs.append(self.apply_cloud_texture)

        if len(p) == 0:
            return img

    def apply_cloud_texture(self, pure_bg, text):
        height = pure_bg.size[1]
        width = pure_bg.size[0]
        noise = self.generate_fractal_noise_2d((height, width),
                                               octaves = int(self.cfg.texture.cloud.octaves),
                                               persistence = float(self.cfg.texture.cloud.persistence),
                                               scale = float(self.cfg.texture.cloud.scale))
        noise = 255 - (((noise + 1) / 2) * 255)
        noise = np.where(noise<130, 30, 205)
        text = np.array(text)
        text[:, :, 3] = noise
        img = Image.alpha_composite(pure_bg, Image.fromarray(text))
        return img.convert('RGB')

    def generate_fractal_noise_2d(self, shape, octaves=2, persistence=0.5, lacunarity = 6.0, scale = 2600.0):
        np_noise = np.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                np_noise[i][j] = noise.pnoise2(i/scale, 
                                            j/scale, 
                                            octaves=octaves, 
                                            persistence=persistence, 
                                            lacunarity=lacunarity,
                                            base=np.random.randint(1,100))
        return np_noise
