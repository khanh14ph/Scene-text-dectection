import torchvision.transforms as transforms
import PIL.ImageEnhance
import PIL.ImageOps
import numpy as np
from skimage import color
from io import BytesIO
from wand.image import Image as WandImage
import cv2
class GaussianBlur:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img
        w, h= img.size
        # kernel = [(31,31)] prev 1 level only
        ksize = int(min(w, h) / 2) // 4
        ksize = (ksize * 2) + 1
        kernel = (ksize, ksize)
        sigmas = [.5, 1, 2]
        if mag < 0 or mag >= len(sigmas):
            index = self.rng.integers(0, len(sigmas))
        else:
            index = mag

        sigma = sigmas[index]
        return transforms.GaussianBlur(kernel_size=kernel, sigma=sigma)(img)
class Contrast:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        # c = [0.4, .3, .2, .1, .05]
        c = [0.4, .3, .2]
        if mag < 0 or mag >= len(c):
            index = self.rng.integers(0, len(c))
        else:
            index = mag
        c = c[index]
        img = np.asarray(img) / 255.
        means = np.mean(img, axis=(0, 1), keepdims=True)
        img = np.clip((img - means) * c + means, 0, 1) * 255

        return Image.fromarray(img.astype(np.uint8))
class VGrid:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, copy=True, max_width=4, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        if copy:
            img = img.copy()
        w, h = img.size

        if mag < 0 or mag > max_width:
            line_width = self.rng.integers(1, max_width)
            image_stripe = self.rng.integers(1, max_width)
        else:
            line_width = 1
            image_stripe = 3 - mag

        n_lines = w // (line_width + image_stripe) + 1
        draw = ImageDraw.Draw(img)
        for i in range(1, n_lines):
            x = image_stripe * i + line_width * (i - 1)
            draw.line([(x, 0), (x, h)], width=line_width, fill='black')

        return img


class HGrid:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, copy=True, max_width=4, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        if copy:
            img = img.copy()
        w, h = img.size
        if mag < 0 or mag > max_width:
            line_width = self.rng.integers(1, max_width)
            image_stripe = self.rng.integers(1, max_width)
        else:
            line_width = 1
            image_stripe = 3 - mag

        n_lines = h // (line_width + image_stripe) + 1
        draw = ImageDraw.Draw(img)
        for i in range(1, n_lines):
            y = image_stripe * i + line_width * (i - 1)
            draw.line([(0, y), (w, y)], width=line_width, fill='black')

        return img

class Grid:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        img = VGrid(self.rng)(img, copy=True, mag=mag, prob=0.7)
        img = HGrid(self.rng)(img, copy=False, mag=mag, prob=0.7)
        return img
class Color:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        c = [.1, .5, .9]
        if mag < 0 or mag >= len(c):
            index = self.rng.integers(0, len(c))
        else:
            index = mag
        c = c[index]
        magnitude = self.rng.uniform(c, c + .6)
        img = PIL.ImageEnhance.Color(img).enhance(magnitude)

        return img

class Rain:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        img = img.copy()
        w, h = img.size
        n_channels = len(img.getbands())
        isgray = n_channels == 1
        line_width = self.rng.integers(1, 2)

        c = [50, 70, 90]
        if mag < 0 or mag >= len(c):
            index = 0
        else:
            index = mag
        c = c[index]

        n_rains = self.rng.integers(c, c + 20)
        slant = self.rng.integers(-60, 60)
        fillcolor = 200 if isgray else (200, 200, 200)

        draw = ImageDraw.Draw(img)
        max_length = min(w, h, 10)
        for i in range(1, n_rains):
            length = self.rng.integers(5, max_length)
            x1 = self.rng.integers(0, w - length)
            y1 = self.rng.integers(0, h - length)
            x2 = x1 + length * math.sin(slant * math.pi / 180.)
            y2 = y1 + length * math.cos(slant * math.pi / 180.)
            x2 = int(x2)
            y2 = int(y2)
            draw.line([(x1, y1), (x2, y2)], width=line_width, fill=fillcolor)

        return img


class Shadow:

    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        # img = img.copy()
        w, h = img.size
        n_channels = len(img.getbands())
        isgray = n_channels == 1

        c = [64, 96, 128]
        if mag < 0 or mag >= len(c):
            index = 0
        else:
            index = mag
        c = c[index]

        img = img.convert('RGBA')
        overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        transparency = self.rng.integers(c, c + 32)
        x1 = self.rng.integers(0, w // 2)
        y1 = 0

        x2 = self.rng.integers(w // 2, w)
        y2 = 0

        x3 = self.rng.integers(w // 2, w)
        y3 = h - 1

        x4 = self.rng.integers(0, w // 2)
        y4 = h - 1

        draw.polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], fill=(0, 0, 0, transparency))

        img = Image.alpha_composite(img, overlay)
        img = img.convert("RGB")
        if isgray:
            img = ImageOps.grayscale(img)

        return img
class Snow:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        w, h = img.size
        c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
             (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
             (0.55, 0.3, 4, 0.9, 12, 8, 0.7)]
        if mag < 0 or mag >= len(c):
            index = self.rng.integers(0, len(c))
        else:
            index = mag
        c = c[index]

        n_channels = len(img.getbands())
        isgray = n_channels == 1

        img = np.asarray(img, dtype=np.float32) / 255.
        if isgray:
            img = np.expand_dims(img, axis=2)
            img = np.repeat(img, 3, axis=2)

        snow_layer = self.rng.normal(size=img.shape[:2], loc=c[0], scale=c[1])  # [:2] for monochrome

        # snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
        snow_layer[snow_layer < c[3]] = 0

        snow_layer = Image.fromarray((np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
        output = BytesIO()
        snow_layer.save(output, format='PNG')
        snow_layer = WandImage(blob=output.getvalue())

        snow_layer.motion_blur(radius=c[4], sigma=c[5], angle=self.rng.uniform(-135, -45))

        snow_layer = cv2.imdecode(np.frombuffer(snow_layer.make_blob(), np.uint8),
                                  cv2.IMREAD_UNCHANGED) / 255.

        # snow_layer = cv2.cvtColor(snow_layer, cv2.COLOR_BGR2RGB)

        snow_layer = snow_layer[..., np.newaxis]

        img = c[6] * img
        gray_img = (1 - c[6]) * np.maximum(img, cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).reshape(h, w, 1) * 1.5 + 0.5)
        img += gray_img
        img = np.clip(img + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255
        img = Image.fromarray(img.astype(np.uint8))
        if isgray:
            img = ImageOps.grayscale(img)

        return img

import numpy as np
from tqdm import tqdm
import shutil
import os
from PIL import Image, ImageOps, ImageDraw
import math
Color=Color()
Grid=Grid()
Snow=Snow()
Rain=Rain()
Shadow=Shadow()
GaussianBlur=GaussianBlur()
Contrast=Contrast()

train_dir="/home/lab/khanhnd/STD_DBNet/dataset/vietnamese/aug_train_images/"
train_gt_dir="/home/lab/khanhnd/STD_DBNet/dataset/vietnamese/aug_train_gts/"
lst=os.listdir(train_dir[:-1])
for i in tqdm(lst):
    im_pt = train_dir+"/"+i
    img_id = im_pt.split("/")[-1].split(".")[0].split("_")[1]
    if int(img_id)<=2000:
        old_img=Image.open(im_pt)
        # for j in range(2,3):
            # new_img=GaussianBlur(old_img,mag=-1, prob=0.3)
        
        new_img=Contrast(old_img, mag=3, prob=0.25)
        new_img=Color(new_img,mag=3, prob=0.3)
        new_img=Grid(new_img,mag=3, prob=0.3)
        new_img=Snow(new_img,mag=3, prob=0.3)
        new_img=Rain(new_img,mag=3, prob=0.3)
        new_img=Shadow(new_img,mag=2, prob=0.3)
        #write new image
        img_filename=train_dir+"img_"+str(int(img_id)+2*1000)+".jpg"
        new_img = new_img.convert('RGB')
        im1 = new_img.save(img_filename)
        old_ann_filename=i+".txt"
        new_ann_filename="img_"+str(int(img_id)+2*1000)+".jpg"+".txt"
        src=train_gt_dir+old_ann_filename
        dst=train_gt_dir+new_ann_filename
        shutil.copyfile(src, dst)
import os
print(len(os.listdir("/home/lab/khanhnd/STD_DBNet/dataset/vietnamese/aug_train_images")))