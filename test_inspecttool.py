from codes.inspect_tool import InspectUI


def main():
    tool = InspectUI(use_camera=False, use_ae=True, use_smooth=True)
    tool.mainloop()


if __name__ == '__main__':
    main()
