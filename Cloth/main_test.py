import numpy as np # must be imported first
from codes.PTCams.FLIRCamera import Camera
import cv2


def main():
    camera = Camera()
    camera.start()

    while True:
        images = camera.get()
        for i, image in enumerate(images):
            scaledW, scaledH = image.shape[1] * 0.5, image.shape[0] * 0.5
            scaledImage = cv2.resize(image, (int(scaledW), int(scaledH)))
            cv2.imshow("image" + str(i), scaledImage)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

    camera.stop()


if __name__ == "__main__":
    main()
