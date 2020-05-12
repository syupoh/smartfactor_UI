try:
    import tkinter as tk
except ImportError:
    import Tkinter as tk

import PIL.Image
import PIL.ImageTk
from .inspect_tool import InspectTool
from .PTCams.FLIRCamera import Camera


class Monitor:
    def __init__(self, window_title):
        self.capsz = 250
        self.butsz = 200
        self.patch_size = 128
        self.delay = 50

        self.cnt = 0

        self.n_defect = 0

        self.window = tk.Tk()
        self.window.title(window_title)

        self.screen_w = self.window.winfo_screenwidth()
        self.screen_h = self.window.winfo_screenheight()
        self.win_w = int(self.screen_w - 150)
        self.win_h = int(self.screen_h - 130)
        self.win_x = int(80)
        self.win_y = int(30)

        self.main_img_w = int((self.win_w - 30) / 3)
        self.main_img_h = int(self.win_h / 2)

        self.window.geometry("{0}x{1}+{2}+{3}".format(self.win_w, self.win_h, self.win_x, self.win_y))

        self.window.resizable(False, False)

        f = open('result.txt', 'w')
        self.resnum = 0
        f.close()

        self.camera = Camera()
        self.camera.start()

        self.main_img = list()
        images = self.camera.get()

        self.detection = InspectTool(self.camera)

        self.main_img1 = PIL.Image.fromarray(images[0])
        self.main_img2 = PIL.Image.fromarray(images[1])
        self.main_img3 = PIL.Image.fromarray(images[2])

        self.buttons = tk.Button(self.window, overrelief="solid",
                                 text='Take',
                                 command=self.click_take, repeatdelay=1000, repeatinterval=100)

        #################

        # self.main_img1 = PIL.Image.open("stitched-leather-texture.jpg")
        # self.main_img2 = PIL.Image.open("stitched-leather-texture.jpg")
        # self.main_img3 = PIL.Image.open("stitched-leather-texture.jpg")

        # self.detected_img1 = PIL.Image.open("stitched-leather-texture_1.png")
        # self.detected_img2 = PIL.Image.open("stitched-leather-texture_2.jpg")
        # self.detected_img3 = PIL.Image.open("stitched-leather-texture_3.jpg")
        # self.detected_img4 = PIL.Image.open("stitched-leather-texture_4.jpg")
        # self.detected_img5 = PIL.Image.open("stitched-leather-texture_5.jpg")
        ##########################################

        self.main_img1 = self.main_img1.resize((self.main_img_w, self.main_img_h))
        self.main_img1_tk = PIL.ImageTk.PhotoImage(self.main_img1)

        self.main_img2 = self.main_img2.resize((self.main_img_w, self.main_img_h))
        self.main_img2_tk = PIL.ImageTk.PhotoImage(self.main_img2)

        self.main_img3 = self.main_img3.resize((self.main_img_w, self.main_img_h))
        self.main_img3_tk = PIL.ImageTk.PhotoImage(self.main_img3)

    def mainloop(self):
        # After it is called once, the update method will be automatically called every delay mill
        self.update()
        self.window.mainloop()

    def update(self):
        self.win_w = self.window.winfo_width()
        self.win_h = self.window.winfo_height()

        self.main_img_w = int((self.win_w - 30) / 3)
        self.main_img_h = int(self.win_h / 2)
        self.main_img_x = int((self.win_w - self.main_img_w * 3 - 20) / 2)
        self.main_img_y = int(10)
        self.patch_y = int(self.win_h - (self.patch_size + 10))

        images = self.camera.get()

        self.main_img1 = PIL.Image.fromarray(images[0])
        self.main_img2 = PIL.Image.fromarray(images[1])
        self.main_img3 = PIL.Image.fromarray(images[2])

        self.buttons.place(y=self.main_img_y + self.main_img_h + 20, x=self.main_img_x, height=40, width=120)

        if (self.main_img_w > 0 and self.main_img_h > 0):
            self.main_img1 = self.main_img1.resize((self.main_img_w, self.main_img_h))
            self.main_img1_tk = PIL.ImageTk.PhotoImage(self.main_img1)
            self.main_lbl1 = tk.Label(self.window, image=self.main_img1_tk)

            self.main_img2 = self.main_img2.resize((self.main_img_w, self.main_img_h))
            self.main_img2_tk = PIL.ImageTk.PhotoImage(self.main_img2)
            self.main_lbl2 = tk.Label(self.window, image=self.main_img2_tk)

            self.main_img3 = self.main_img3.resize((self.main_img_w, self.main_img_h))
            self.main_img3_tk = PIL.ImageTk.PhotoImage(self.main_img3)
            self.main_lbl3 = tk.Label(self.window, image=self.main_img3_tk)

        self.size_defect = "5 2 3 6 2"
        self.dist_defect = "2 3 1 5 6"
        ####################

        # detected_img1 = self.detected_img1.resize((self.patch_size, self.patch_size))
        # detected_img2 = self.detected_img2.resize((self.patch_size, self.patch_size))
        # detected_img3 = self.detected_img3.resize((self.patch_size, self.patch_size))
        # detected_img4 = self.detected_img4.resize((self.patch_size, self.patch_size))
        # detected_img5 = self.detected_img5.resize((self.patch_size, self.patch_size))
        #
        # detected_img1_tk = PIL.ImageTk.PhotoImage(detected_img1)
        # detected_img2_tk = PIL.ImageTk.PhotoImage(detected_img2)
        # detected_img3_tk = PIL.ImageTk.PhotoImage(detected_img3)
        # detected_img4_tk = PIL.ImageTk.PhotoImage(detected_img4)
        # detected_img5_tk = PIL.ImageTk.PhotoImage(detected_img5)

        # self.detected_lbl1 = tk.Label(self.window, image=detected_img1_tk)
        # self.detected_lbl2 = tk.Label(self.window, image=detected_img2_tk)
        # self.detected_lbl3 = tk.Label(self.window, image=detected_img3_tk)
        # self.detected_lbl4 = tk.Label(self.window, image=detected_img4_tk)
        # self.detected_lbl5 = tk.Label(self.window, image=detected_img5_tk)
        #
        # self.detected_lbl1.place(x=self.main_img_x, y=self.patch_y, width=self.patch_size, height=self.patch_size)
        # self.detected_lbl2.place(x=self.main_img_x+(self.patch_size+10)*1, y=self.patch_y, width=self.patch_size, height=self.patch_size)
        # self.detected_lbl3.place(x=self.main_img_x+(self.patch_size+10)*2, y=self.patch_y, width=self.patch_size, height=self.patch_size)
        # self.detected_lbl4.place(x=self.main_img_x+(self.patch_size+10)*3, y=self.patch_y, width=self.patch_size, height=self.patch_size)
        # self.detected_lbl5.place(x=self.main_img_x+(self.patch_size+10)*4, y=self.patch_y, width=self.patch_size, height=self.patch_size)

        self.main_lbl1 = tk.Label(self.window, image=self.main_img1_tk)
        self.main_lbl2 = tk.Label(self.window, image=self.main_img2_tk)
        self.main_lbl3 = tk.Label(self.window, image=self.main_img3_tk)

        self.main_lbl1.place(x=self.main_img_x, y=self.main_img_y, width=self.main_img_w, height=self.main_img_h)
        self.main_lbl2.place(x=self.main_img_x + (self.main_img_w + 10) * 1, y=self.main_img_y, width=self.main_img_w,
                             height=self.main_img_h)
        self.main_lbl3.place(x=self.main_img_x + (self.main_img_w + 10) * 2, y=self.main_img_y, width=self.main_img_w,
                             height=self.main_img_h)

        self.label1 = tk.Label(self.window, text="defect 갯수 {0}".format(self.n_defect), height=5)
        # self.label2 = tk.Label(self.window, text="defect 크기 {0}".format(self.size_defect), height=5)
        # self.label3 = tk.Label(self.window, text="defect 간 거리 {0}".format(self.dist_defect), height=5)

        self.label1.place(x=self.main_img_x + (self.patch_size + 10) * 5 + 10, y=self.patch_y, height=13)
        # self.label2.place(x=self.main_img_x+(self.patch_size+10)*5+10, y=self.patch_y+(13+5)*1, height=13)
        # self.label3.place(x=self.main_img_x+(self.patch_size+10)*5+10, y=self.patch_y+(13+5)*2, height=13)

        self.window.after(self.delay, self.update)

        self.cnt += 1
        if self.cnt == 2:
            self.click_take()
            self.cnt = 0

    def click_take(self):
        self.detection.OnClick_take()
        self.n_defect, ins_patch, text_patch = self.detection.OnClick_inspect()

        self.resnum += 1
        for i, patch in enumerate(ins_patch):
            # if i == 0:
            #     self.detected_img1 = patch
            # if i == 1:

            #     self.detected_img2 = patch
            # if i == 2:
            #     self.detected_img3 = patch
            # if i == 3:
            #     self.detected_img4 = patch
            # if i == 4:
            #     self.detected_img5 = patch
            patch.save("detected_patch2/detected_patch{0}_{1}_{2}.png".format(self.resnum, i, text_patch[i]))
        f = open('result.txt', 'a')
        f.write("inspection {0}\n".format(self.resnum))
        f.write("\tdefect  갯수 {0}\n".format(self.n_defect))
        for i, text in enumerate(text_patch):
            f.write("\tdefect {0} : {1}\n".format(i, text))

        # f.write("\tdefect  크기 {0}\n".format(self.size_defect))
        # f.write("\tdefect  갯수 {0}\n".format(self.n_defect))

        f.close()
