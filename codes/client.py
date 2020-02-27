import socket
import pickle
import json
import numpy as np
import tensorflow as tf
import time
from codes.config import *
from .ae import Detector

from tkinter import *
from PIL import ImageTk, Image
from imageio import imread, imsave
import npy
from .config import SERVER_HOST, SERVER_PORT
from .socket_funcs import receive, send
from .utils import task, Dummy
from .PTCams.FLIRCamera import Camera

import pdb

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
    def __init__(self, camera, use_camera=True):
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

        self.inspect_result = []

        self.window.geometry("%dx%d+200+100" % (1800, 1000))
        self.window.resizable(False, False)

        self.buttons = Dummy()
        self.labels = Dummy()
        self.frames = Dummy()

        self.use_camera = use_camera

        # if use_camera:
        #     self.camera = Camera()
        #     self.camera.start()
        # else:
        #     self.camera = None
        self.camera = camera

        self._sess = None
        self._detector = None

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
                    self.thres = v / 10000
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
        # print('Using test image!')
        if self.use_camera:
            images = self.camera.get()
            image = self.merge_images(images)
            image = image[700:, 200: -1000]
            #print(image.shape)
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
        return self._display_inspect_result(coords, defect_probs)

    def _clear_inspect_result(self):
        for v in self.inspect_result:
            self.canvas_image.delete(v)
        self.inspect_result.clear()

    def _display_inspect_result(self, coords, defect_probs):
        s = self.scale
        n_defect = 0
        ins_patch = list()
        text_patch = list()

        oriimg = Image.fromarray(self.img_original)
        # oriimg2 = self.img

        for prob, coords in zip(defect_probs, coords):
            if prob > self.thres:
                j, i = coords
                area = (i, j, i+128, j+128)
                ins_patch.append(oriimg.crop(area))
                # ins_patch.append(oriimg2[(i*s):(i+128)*s, (j*s):(j+128)*s, :])
                r = self.canvas_image.create_rectangle(i * s, j * s, (i + 128) * s, (j + 128) * s,
                                                   fill="", width=3, outline='red')

                x, y = pixel_to_cm(i, j)
                text = '(%.1f, %.1f)' % (x, y)
                text_patch.append(text)
                t = self.canvas_image.create_text((i + 64) * s, (j + 128 + 20) * s, fill="red",
                                              font="Helvetica 9", text=text)
                self.inspect_result.append(r)
                self.inspect_result.append(t)

                n_defect += 1
        # pdb.set_trace()
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


class PiHub(InspectTool):
    def __init__(self):
        super().__init__()

    @property
    def leftCameraConfig(self):
        return {
            'awb_mode': 'auto',
            'exposure_mode': 'auto',
            'flash_mode': 'auto',
            'brightness': 45,
            'contrast': 99,
            'saturation': 70,
        }

    @property
    def rightCameraConfig(self):
        return {
            'awb_mode': 'auto',
            'exposure_mode': 'auto',
            'flash_mode': 'on',

            'brightness': 46,
            'contrast': 64,
            'saturation': 12,
        }

    def get_image(self):
        camera1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        camera2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        camera1.connect((CAMERA1_HOST, CAMERA1_PORT))
        camera2.connect((CAMERA2_HOST, CAMERA2_PORT))

        image1_file = "left_camera.jpg"
        image2_file = "right_camera.jpg"

        # 1. Tx JPEG
        packet1 = self.leftCameraConfig
        packet2 = self.rightCameraConfig
        serial1 = json.dumps(packet1)
        serial2 = json.dumps(packet2)
        camera1.send(serial1.encode("utf-8"))
        camera2.send(serial2.encode("utf-8"))

        # 2. Rx SIZE size
        packet1 = camera1.recv(1024)
        packet2 = camera2.recv(1024)
        print("packet from camera1 = ", packet1.decode("ascii"))
        print("packet from camera2 = ", packet2.decode("ascii"))
        message1 = packet1.decode("ascii")
        message2 = packet2.decode("ascii")
        tmp1 = message1.split()
        tmp2 = message2.split()
        image1_size = int(tmp1[1])
        image2_size = int(tmp2[1])

        # 3. Tx SIZE size
        packet1 = "REQ TX %s" % image1_size
        packet2 = "REQ TX %s" % image2_size
        camera1.send(packet1.encode("ascii"))
        camera2.send(packet2.encode("ascii"))

        # 4. Rx image data
        # image1_file = "/run/user/1000/camera1.jpg"
        # image2_file = "/run/user/1000/camera2.jpg"
        file1 = open(image1_file, "wb")
        file2 = open(image2_file, "wb")

        total_rx1 = 0
        total_rx2 = 0
        while (total_rx1 < image1_size) or (total_rx2 < image2_size):
            packet1 = b''
            if (total_rx1 < image1_size):
                packet1 = camera1.recv(4 * 1024)
                total_rx1 += len(packet1)
                if len(packet1):
                    # print('size = ', len(packet1))
                    file1.write(packet1)
            # print("total rx1 = ", total_rx1)
            packet2 = b''
            if (total_rx2 < image2_size):
                packet2 = camera2.recv(4 * 1024)
                total_rx2 += len(packet2)
                if len(packet2):
                    # print('size = ', len(packet1))
                    file2.write(packet2)
                # print("total rx2 = ", total_rx2)

            if (len(packet1) == 0) and (len(packet2) == 0):
                break

        file1.close()
        file2.close()

        camera1.shutdown(socket.SHUT_RDWR)
        camera2.shutdown(socket.SHUT_RDWR)
        camera1.close()
        camera2.close()
        image = imread(image2_file)

        image = image[56:]
        H, W = image.shape[:2]
        image = npy.image.resize(image, (H * 2, W * 2))
        image = image[:, 128 * 3: -128 * 3]
        return image


class Client(PiHub):
    def __init__(self):
        self.host = SERVER_HOST
        self.port = SERVER_PORT
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        super().__init__()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.socket.close()

    def sendObj(self, obj):
        data = pickle.dumps(obj)
        send(self.socket, data)

    def receive(self):
        data = receive(self.socket, PACKET_SIZE)
        obj = pickle.loads(data)
        return obj

    def inspect(self, patches):
        self.socket.connect((self.host, self.port))
        self.sendObj(patches)
        result = self.receive()
        self.socket.close()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        return result
