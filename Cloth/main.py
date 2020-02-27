from codes.client import InspectTool
from npy import set_cuda


def main():
    app = InspectTool(use_camera=False)
    app.run()


if __name__ == '__main__':
    set_cuda(2)
    main()
