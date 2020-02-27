import numpy as np
import PySpin
import multiprocessing as mp



def acquisition_mp(cam, idx):
    import PySpin
    # cam = self.__cam_list.GetByIndex(idx)
    node_acquisition_mode = PySpin.CEnumerationPtr(cam.GetNodeMap().GetNode('AcquisitionMode'))
    if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
        print('Unable to set acquisition mode to continuous (node retrieval; camera %d). Aborting... \n'
              % idx, flush=True)
        return False

    node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
    if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(
            node_acquisition_mode_continuous):
        print('Unable to set acquisition mode to continuous (entry \'continuous\' retrieval %d). \
        Aborting... \n' % idx, flush=True)
        return False

    acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()
    node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

    cam.BeginAcquisition()
    print('Begin Acquisition', flush=True)
    while True:
        image_result = cam.GetNextImage()
        if not image_result.IsIncomplete():
            print('image updated', idx, flush=True)
            width = image_result.GetWidth()
            height = image_result.GetHeight()
            image_converted = image_result.Convert(PySpin.PixelFormat_BGR8, PySpin.DEFAULT)
            np_image = np.array(image_converted.GetData())
            np_image = np_image.reshape((height, width, 3))
            # self.__image_list[idx] = np_image


class FLIRCameraHandler:
    def __init__(self):
        self.__system = PySpin.System.GetInstance()
        self.__version = self.__system.GetLibraryVersion()
        self.__versionStr = 'Spinnaker Library version: %d.%d.%d.%d' % \
                            (self.__version.major, self.__version.minor, self.__version.type, self.__version.build)
        self.__cam_list = self.__system.GetCameras()
        self.num_cameras = self.__cam_list.GetSize()
        self.is_valid = True
        if self.num_cameras == 0:
            self.__cam_list.Clear()
            self.__system.ReleaseInstance()
            self.is_valid = False
            pass
        self.__pool = mp.Pool(processes=self.num_cameras)
        self.__image_list = mp.Manager().list()
        for i, cam in enumerate(self.__cam_list):
            cam.Init()
            self.__image_list.append(np.zeros((100, 100, 3)))
        self.__raw_images = None
        pass

    def __del__(self):
        if self.__cam_list:
            for cam in self.__cam_list:
                if cam.IsStreaming():
                    cam.EndAcquisition()
                cam.DeInit()
            self.__cam_list.Clear()
        if self.__system:
            self.__system.ReleaseInstance()

    def start_acquisition(self, mode, width, height, exposure_time):
        for idx, cam in enumerate(self.__cam_list):
            # node_acquisition_mode = PySpin.CEnumerationPtr(cam.GetNodeMap().GetNode('AcquisitionMode'))
            # if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
            #     print('Unable to set acquisition mode to continuous (node retrieval; camera %d). Aborting... \n'
            #           % idx, flush=True)
            #     return False
            #
            # node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
            # if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(
            #         node_acquisition_mode_continuous):
            #     print('Unable to set acquisition mode to continuous (entry \'continuous\' retrieval %d). \
            #         Aborting... \n' % idx, flush=True)
            #     return False
            #
            # acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()
            # node_acquisition_mode.SetIntValue(acquisition_mode_continuous)
            nodemap = cam.GetNodeMap()
            ### Set Pixel Format to RGB8 ###
            node_pixel_format = PySpin.CEnumerationPtr(nodemap.GetNode('PixelFormat'))

            if not PySpin.IsAvailable(node_pixel_format) or not PySpin.IsWritable(node_pixel_format):
                print('Unable to set Pixel Format to '+str(mode)+' (enum retrieval). Aborting...')
            else:
                node_pixel_format_val = node_pixel_format.GetEntryByName(mode)
                if not PySpin.IsAvailable(node_pixel_format_val) or not PySpin.IsReadable(node_pixel_format_val):
                    print('Unable to set Pixel Format to '+str(mode)+' (entry retrieval). Aborting...')
                else:
                    pixel_format_val = node_pixel_format_val.GetValue()
                    node_pixel_format.SetIntValue(pixel_format_val)

            # TODO: exception handling...
            s_node_map = cam.GetTLStreamNodeMap()
            handling_mode = PySpin.CEnumerationPtr(s_node_map.GetNode('StreamBufferHandlingMode'))
            handling_mode_entry = handling_mode.GetEntryByName('NewestOnly')
            handling_mode.SetIntValue(handling_mode_entry.GetValue())

            node_width_max = PySpin.CIntegerPtr(nodemap.GetNode('WidthMax'))
            if width < 0:
                w = node_width_max.GetValue()
            else:
                w = width

            node_height_max = PySpin.CIntegerPtr(nodemap.GetNode('HeightMax'))
            if height < 0:
                h = node_height_max.GetValue()
            else:
                h = height

            node_iWidth_int = PySpin.CIntegerPtr(nodemap.GetNode("Width"))
            if not PySpin.IsAvailable(node_iWidth_int) or not PySpin.IsWritable(node_iWidth_int):
                print('Unable to set width')
            else:
                node_iWidth_int.SetValue(w)

            node_iHeight_int = PySpin.CIntegerPtr(nodemap.GetNode("Height"))
            if not PySpin.IsAvailable(node_iHeight_int) or not PySpin.IsWritable(node_iHeight_int):
                print('Unable to set height')
            else:
                node_iHeight_int.SetValue(h)

            node_iTimestampLatch_cmd = PySpin.CCommandPtr(nodemap.GetNode("TimestampLatch"))
            if node_iTimestampLatch_cmd is not None:
                # Execute command
                print('timestamp latching...')
                node_iTimestampLatch_cmd.Execute()

            node_exposure_auto = PySpin.CEnumerationPtr(nodemap.GetNode('ExposureAuto'))
            if not PySpin.IsAvailable(node_exposure_auto) or not PySpin.IsWritable(node_exposure_auto):
                print('unable to set exposure auto')
            else:
                if exposure_time < 0:
                    node_exposure_auto_val = node_exposure_auto.GetEntryByName('Continuous')
                    if not PySpin.IsAvailable(node_exposure_auto_val) or not PySpin.IsWritable(node_exposure_auto_val):
                        print('unable to set exposure auto to continuous')
                    else:
                        node_exposure_auto.SetIntValue(node_exposure_auto_val.GetValue())
                else:
                    node_exposure_auto_val = node_exposure_auto.GetEntryByName('Off')
                    if not PySpin.IsAvailable(node_exposure_auto_val) or not PySpin.IsWritable(node_exposure_auto_val):
                        print('unable to set exposure auto to off')
                    else:
                        node_exposure_auto.SetIntValue(node_exposure_auto_val.GetValue())

            node_exposure_mode = PySpin.CEnumerationPtr(nodemap.GetNode('ExposureMode'))
            if not PySpin.IsAvailable(node_exposure_mode) or not PySpin.IsWritable(node_exposure_mode):
                print('unable to set exposure mode')
            else:
                node_exposure_mode_val = node_exposure_mode.GetEntryByName('Timed')
                if not PySpin.IsAvailable(node_exposure_mode_val) or not PySpin.IsWritable(node_exposure_mode_val):
                    print('unable to set exposure mode timed')
                else:
                    node_exposure_mode.SetIntValue(node_exposure_mode_val.GetValue())

            #TODO: check min and max before set exposure time
            if exposure_time > 0:
                exposure_time_ptr = PySpin.CFloatPtr(nodemap.GetNode('ExposureTime'))
                if not PySpin.IsAvailable(exposure_time_ptr) or not PySpin.IsWritable(exposure_time_ptr):
                    print('unable to set exposure time')
                else:
                    exposure_time_ptr.SetValue(exposure_time)
                    print('set exposure time')

            cam.BeginAcquisition()
        return

    def grab(self):
        if self.__raw_images is None:
            self.__raw_images = []
            for idx, cam in enumerate(self.__cam_list):
                image_result = cam.GetNextImage()
                if not image_result.IsIncomplete():
                    width = image_result.GetWidth()
                    height = image_result.GetHeight()
                    # image_converted = image_result.Convert(PySpin.PixelFormat_BGR8, PySpin.NEAREST_NEIGHBOR )
                    # self.__raw_images.append(image_converted)
                    self.__raw_images.append(image_result)
        else:
            for idx, cam in enumerate(self.__cam_list):
                image_result = cam.GetNextImage()
                if not image_result.IsIncomplete():
                    width = image_result.GetWidth()
                    height = image_result.GetHeight()
                    # image_converted = image_result.Convert(PySpin.PixelFormat_BGR8, PySpin.NEAREST_NEIGHBOR )
                    # self.__raw_images[idx] = image_converted
                    self.__raw_images[idx] = image_result
        i = int(0)
        for image in self.__raw_images:
            # np_image = np.array(image.GetData())
            # np_image = np_image.reshape((image.GetHeight(), image.GetWidth(), 3))
            ##print('camera #%d image updated at %fms' % (i, image.GetTimeStamp() / (1000 * 1000)))
            self.__image_list[i] = image.GetNDArray()
            #image.Release()
            i += 1

        return

    def get_image(self, idx):
        return self.__image_list[idx]

    def stop_acquisition(self):
        self.__pool.close()
        self.__pool.join()
        return

    def get_version_info(self):
        return self.__versionStr


