from codes.inspect_tool import InspectUI


def main():
    tool = InspectUI(use_camera=False, use_inspect=False)
    tool.mainloop()


if __name__ == '__main__':
    main()
