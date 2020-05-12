import numpy as np
import tensorflow as tf
import time
from .ae import Detector

from tkinter import *
from PIL import ImageTk, Image
from imageio import imread, imsave
import npy
from .utils import task, Dummy


def image_to_patches(image):
    H, W = image.shape[:2]
    patches = list()
    coords = list()
    P = 128
    for y in range(0, H, P):
        for x in range(0, W, P):
            y = min(y, H - P)
            x = min(x, W - P)
            patch = image[y: y + P, x: x + P]
            coord = (y, x)
            patches.append(patch.copy())
            coords.append(coord)
    return np.asarray(patches), np.asarray(coords)


def pixel_to_cm(i, j):
    x = (i - 1536) * 70 / 3840
    y = 4 + j * 33 / 2160
    return x, y


class PatchInspectCore:
    def __init__(self, use_ae):
        self.use_ae = use_ae

        self._sess = None
        self._detector = None

    @property
    def sess(self):
        if self._sess is not None:
            return self._sess
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self._sess = tf.Session(config=config)
        return self._sess

    @property
    def detector(self):
        if self._detector is not None:
            return self._detector
        self._detector = Detector(self.sess, d=8)
        self._detector.load()
        return self._detector

    # Inspect logics

    def inspect_local(self, patches) -> np.ndarray:  # [N_patch]
        patches = npy.image.rgb2gray(patches, keep_dims=True)
        patches = npy.image.to_float32(patches)
        recons = self.detector.recon(patches)
        residual = np.square(patches - recons)
        scores = residual.max(axis=-1).max(axis=-1).max(axis=-1)

        return scores

    def inspect_random(self, patches) -> np.ndarray:  # [N_patch]
        print('Showing dummy result!')
        N = patches.shape[0]
        result = np.random.uniform(0, 1, size=N)
        return result

    def inspect(self, patches):
        if self.use_ae:
            result = self.inspect_local(patches)
        else:
            result = self.inspect_random(patches)

        return result


class InspectResultEntry:
    def __init__(self, x, y, h, w, p):
        self.x = x
        self.y = y
        self.h = h
        self.w = w
        self.p = p


class InspectUI(Frame):
    def __init__(self, camera=None, use_camera=True, use_ae=True):
        self.window = Toplevel()
        self.window.title('Inspection')
        self.swit = 0

        self.scale = 1.
        self.img_original = None
        self.img = None
        self.img_tk = None
        self.img_created = None
        self.dir_path = ''
        self.img_path = ''
        self.img_path_list = []

        self.xoffset = 50
        self.yoffset = 30
        self.patch_size = 128
        self.thres = 0.5

        self.inspect_display_components = []

        self.window.geometry("%dx%d+200+100" % (1800, 1000))
        self.window.resizable(False, False)

        self.buttons = Dummy()
        self.labels = Dummy()
        self.frames = Dummy()

        self.use_camera = use_camera
        self.use_ae = use_ae
        self.inspect_core = PatchInspectCore(use_ae)
        self.camera = camera

        with task('Set layout'):
            with task('Set frames'):
                self.frames.menu = Frame(self.window, relief='solid', width=200, height=200, bg='white')
                self.frames.menu.grid(row=0, column=0)

                self.frames.history = Frame(self.window, relief='solid', width=200, height=800, bg='gray')
                self.frames.history.grid(row=1, column=0)

                self.frames.image = Frame(self.window, relief='solid', width=800, height=800, bg='black')
                self.frames.image.grid(row=0, column=1, rowspan=2)
                self.canvas_image = Canvas(self.frames.image, width=1600, height=1000, bd=0, highlightthickness=0)
                self.canvas_image.pack()

            with task('Add widgets'):
                # 1. make buttons
                self.buttons.take = Button(self.frames.menu, overrelief="solid",
                                           text='Take',
                                           command=self.OnClick_take, repeatdelay=1000, repeatinterval=100)

                self.buttons.inspect = Button(self.frames.menu, overrelief="solid",
                                              text='Inspect',
                                              command=self.OnClick_inspect, repeatdelay=1000, repeatinterval=100)

                # 2. place buttons
                self.buttons.inspect.place(y=self.yoffset + 120, x=50, height=40, width=120)
                self.buttons.take.place(y=self.yoffset + 60, x=50, height=40, width=120)

            with task():
                def select(_):
                    v = int(self.labels.slider.get())
                    self.thres = v / 100
                    self.labels.message.config(text='Threshold: %d' % v)

                # 1. make labels and texts
                self.labels.message = Label(self.window, text='Threshold: 0', bg='white')
                self.thres_var = IntVar()
                self.labels.slider = Scale(self.window, variable=self.thres_var, command=select, orient='horizontal',
                                           showvalue=False, tickinterval=25, to=100, length=180)
                self.labels.slider.set(90)
                self.labels.slider.place(y=self.yoffset + 20, x=10)

                # 2. place labels and texts
                self.labels.message.place(y=self.yoffset, x=50)

        super().__init__()

    def merge_images(self, images):
        return np.concatenate(images, axis=1)

    def get_image(self):
        if self.use_camera:
            images = self.camera.get()
            image = self.merge_images(images)
            image = image[700:, 200: -1000]

        else:
            print('Using test image!')
            image = imread('data/191101/test.png')

        return image

    def OnClick_take(self):
        image = self.get_image()
        self.set_image(image)

    def inspect_and_get_result(self, image):
        patches, coords = image_to_patches(image)
        s = time.time()
        print('Inspect start.')
        defect_probs = self.inspect_core.inspect(patches)
        e = time.time()
        print('Inspect done. Took %.2fs' % (e - s))

        results = list()
        for (y, x), defect_prob in zip(coords, defect_probs):
            result = InspectResultEntry(x, y, 128, 128, defect_prob)
            results.append(result)
        return results

    def OnClick_inspect(self):
        self._clear_inspect_result()
        inspect_results = self.inspect_and_get_result(self.img_original)
        return self._display_inspect_result(inspect_results)

    def _clear_inspect_result(self):
        for v in self.inspect_display_components:
            self.canvas_image.delete(v)
        self.inspect_display_components.clear()

    def _display_inspect_result(self, inspect_results):
        s = self.scale
        n_defect = 0
        ins_patch = list()
        text_patch = list()

        oriimg = Image.fromarray(self.img_original)
        # oriimg2 = self.img

        for r in inspect_results:
            if r.p > self.thres:
                area = (r.x, r.y, r.x + r.w, r.y + r.h)
                ins_patch.append(oriimg.crop(area))
                rect = self.canvas_image.create_rectangle(
                    r.x * s, r.y * s,
                    (r.x + r.w) * s, (r.y + r.h) * s,
                    fill="", width=3, outline='red'
                )

                x, y = pixel_to_cm(r.x, r.y)
                text = '(%.1f, %.1f)' % (x, y)
                text_patch.append(text)

                text = self.canvas_image.create_text(
                    (r.x + 64) * s, (r.y + r.h + 20) * s,
                    fill="red", font="Helvetica 9", text=text
                )
                self.inspect_display_components.append(rect)
                self.inspect_display_components.append(text)

                n_defect += 1

        return n_defect, ins_patch, text_patch

    def set_image(self, image):
        self.img_original = image
        image = self._resize_image(image)
        self._display_image(image)

    def _resize_image(self, image):
        max_h = 1000
        max_w = 1600

        H, W = image.shape[:2]
        self.scale = min(1, max_h / H, max_w / W)

        h = int(H * self.scale)
        w = int(W * self.scale)
        image = npy.image.resize(image, (h, w))
        print('Image reshaped from %s to %s. scale: %.3f' % ((H, W), (h, w), self.scale))
        return image

    def _display_image(self, image):
        self.img = image
        self.img_tk = ImageTk.PhotoImage(Image.fromarray(image))
        if self.img_created is not None:
            self.canvas_image.delete(self.img_created)

        self.img_created = self.canvas_image.create_image(0, 0, anchor="nw", image=self.img_tk)

    ###########

    def run(self):
        self.window.mainloop()
