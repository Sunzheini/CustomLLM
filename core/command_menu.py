class CommandMenu:
    """
    A simple command-line menu system that maps user commands to functions.
    Users can enter commands to execute corresponding functions or 'q' to quit.
    """
    def __init__(self, commands: dict=None):
        if commands is None:
            self.commands = {}
        else:
            self.commands = commands

    def run(self) -> None:
        """
        Runs the command menu, prompting the user for input and executing corresponding functions.
        Users can type 'q' to quit the program.
        :return: None
        """
        while 1:
            user_command = input("Enter command (type 'q' to quit): ")

            if user_command == 'q':
                print("Exiting the program.")
                break

            elif user_command in self.commands:
                try:
                    self.commands[user_command]()
                except Exception as e:
                    print(f"Error executing command '{user_command}': {e}")

            else:
                print(f"Unknown command: {user_command}. Please try again.")
