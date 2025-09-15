from core.command_menu import CommandMenu


def function_1():
    print('Start')


if __name__ == "__main__":
    menu = CommandMenu({
        '1': function_1,
    })
    menu.run()
