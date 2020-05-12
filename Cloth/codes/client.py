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
    for i in range(0, H, P):
        for j in range(0, W, P):
            i = min(i, H - P)
            j = min(j, W - P)
            patch = image[i: i + P, j: j + P]
            coord = (i, j)
            patches.append(patch.copy())
            coords.append(coord)
    return np.asarray(patches), np.asarray(coords)


def pixel_to_cm(i, j):
    x = (i - 1536) * 70 / 3840
    y = 4 + j * 33 / 2160
    return x, y


class InspectTool(Frame):
    def __init__(self, use_camera=True):
        self.window = Tk()
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

        self.inspect_result = []

        self.window.geometry("%dx%d+200+100" % (1800, 1000))
        self.window.resizable(False, False)

        self.buttons = Dummy()
        self.labels = Dummy()
        self.frames = Dummy()

        self.use_camera = use_camera
        if use_camera:
            from .PTCams.FLIRCamera import Camera
            self.camera = Camera()
            self.camera.start()
        else:
            self.camera = None

        self._sess = None
        self._detector = None

        with task('Set layout'):
            with task('Set frames'):
                self.frames.menu = Frame(self.window, relief='solid', width=200, height=300, bg='white')
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

                self.buttons.take_and_inspect = Button(self.frames.menu, overrelief="solid",
                                                       text='Take & Inspect',
                                                       command=self.OnClick_take_and_inspect, repeatdelay=1000,
                                                       repeatinterval=100)

                # 2. place buttons
                self.buttons.inspect.place(y=self.yoffset + 120, x=50, height=40, width=120)
                self.buttons.take.place(y=self.yoffset + 60, x=50, height=40, width=120)
                self.buttons.take_and_inspect.place(y=self.yoffset + 180, x=50, height=40, width=120)

            with task():
                def select(_):
                    v = int(self.labels.slider.get())
                    self.thres = v / 10000
                    self.labels.message.config(text='Threshold: %d' % v)

                # 1. make labels and texts
                self.labels.message = Label(self.window, text='Threshold: 0', bg='white')
                self.thres_var = IntVar()
                self.labels.slider = Scale(self.window, variable=self.thres_var, command=select, orient='horizontal',
                                           showvalue=False, tickinterval=25, to=100, length=180)
                self.labels.slider.place(y=self.yoffset + 20, x=10)

                # 2. place labels and texts
                self.labels.message.place(y=self.yoffset, x=50)

        super().__init__()

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

    def merge_images(self, images):
        return np.concatenate(images, axis=1)

    def get_image(self):
        print('Using test image!')
        if self.use_camera:
            images = self.camera.get()
            image = self.merge_images(images)
        else:
            image = imread('data/191101/test.png')

        return image

    def inspect_local(self, patches):
        patches = npy.image.rgb2gray(patches, keep_dims=True)
        patches = npy.image.to_float32(patches)
        recons = self.detector.recon(patches)
        residual = np.square(patches - recons)
        scores = residual.max(axis=-1).max(axis=-1).max(axis=-1)

        return scores

    def inspect_random(self, patches):
        print('Showing dummy result!')
        N = patches.shape[0]
        result = np.random.uniform(0, 1, size=N)
        return result

    def inspect(self, patches):
        result = self.inspect_local(patches)
        return result

    def OnClick_take(self):
        image = self.get_image()
        self.set_image(image)

    def OnClick_inspect(self):
        self._clear_inspect_result()
        patches, coords = image_to_patches(self.img_original)
        s = time.time()
        print('Inspect start.')
        defect_probs = self.inspect(patches)
        e = time.time()
        print('Inspect done. Took %.2fs' % (e - s))
        self._display_inspect_result(coords, defect_probs)

    def OnClick_take_and_inspect(self):
        self.OnClick_take()
        self.OnClick_inspect()

    def _clear_inspect_result(self):
        for v in self.inspect_result:
            self.canvas_image.delete(v)
        self.inspect_result.clear()

    def _display_inspect_result(self, coords, defect_probs):
        s = self.scale
        for prob, coords in zip(defect_probs, coords):
            if prob > self.thres:
                j, i = coords
                r = self.canvas_image.create_rectangle(i * s, j * s, (i + 128) * s, (j + 128) * s,
                                                       fill="", width=3, outline='red')

                x, y = pixel_to_cm(i, j)
                text = '(%.1f, %.1f)' % (x, y)

                t = self.canvas_image.create_text((i + 64) * s, (j + 128 + 20) * s, fill="red",
                                                  font="Helvetica 9", text=text)
                self.inspect_result.append(r)
                self.inspect_result.append(t)

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
