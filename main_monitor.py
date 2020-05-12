from codes.monitor import Monitor


def main():
    import npy
    npy.set_tf_log()
    app = Monitor("Smart Factory Demo")
    app.mainloop()


if __name__ == "__main__":
    main()
