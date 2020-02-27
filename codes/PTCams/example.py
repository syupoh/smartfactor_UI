import numpy as np # must be imported first
import FLIRCameraHandler
import cv2
import time





def main():
    handler = FLIRCameraHandler.FLIRCameraHandler()
    print(handler.get_version_info())  # check library is imported correctly
    print("Connected Device: ", handler.num_cameras)  # check library is imported correctly

    handler.start_acquisition('BGR8', -1, -1, 1000)
    saveFlag = False
    saveCnt = 0
    while True:
        start = time.time()
        handler.grab()
        for i in range(handler.num_cameras):
            image = handler.get_image(i)
            scaledW, scaledH = image.shape[1] * 0.5, image.shape[0] * 0.5
            scaledImage = cv2.resize(image, (int(scaledW), int(scaledH)))
            # scaledImage = cv2.cvtColor(scaledImage, cv2.COLOR_RGB2BGR);
            cv2.imshow("image" + str(i), scaledImage)
            if saveFlag:
                cv2.imwrite("cam{0}_img{1}.png".format(i, saveCnt), image)
        if saveFlag:
            saveCnt += 1
            if saveCnt >= 30:
                saveFlag = False

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            saveFlag = True

        end = time.time()
        seconds = end - start
        print("Time taken : {0} seconds".format(seconds))
        fps = float(1.0) / seconds
        print("fps: {0}".format(fps))

    handler.stop_acquisition()


if __name__ == "__main__":
    main()
