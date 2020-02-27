import numpy as np
from . import FLIRCameraHandler
import time


class Camera:
    def __init__(self, print_times=True):
        self.handler = FLIRCameraHandler.FLIRCameraHandler()
        self.print_times = print_times
        print(self.handler.get_version_info())  # check library is imported correctly
        print("Connected Device: ", self.handler.num_cameras)  # check library is imported correctly

    def start(self):
        self.handler.start_acquisition('BGR8', -1, -1, 1000)

    def get(self):
        start = time.time()
        self.handler.grab()
        images = list()
        for i in range(self.handler.num_cameras):
            image = self.handler.get_image(i)
            image = image[:, :, [2, 1, 0]]
            images.append(image)
        end = time.time()
        seconds = end - start

        if self.print_times:
            print("Time taken : {0} seconds".format(seconds))
        return images

    def stop(self):
        self.handler.stop_acquisition()

