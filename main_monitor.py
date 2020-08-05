from codes.monitor import Monitor
from codes.utils import set_tf_log


def main():
    set_tf_log()
    app = Monitor("Smart Factory Demo")
    app.mainloop()


if __name__ == "__main__":
    main()
