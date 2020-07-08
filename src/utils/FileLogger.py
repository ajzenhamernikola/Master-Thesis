import os


class FileLogger(object):
    def __init__(self, file_name, print_to_stdout=True):
        self.file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models",
                                                      f"report_{file_name}.txt"))
        self.file = None
        self.print_to_stdout = print_to_stdout

        # Checks
        if os.path.exists(self.file_path):
            print(f"The file {self.file_path} already exists. Overwrite? [y/n]")
            overwrite = input().lower() == "y"
            if not overwrite:
                self.file = open(self.file_path, "a")
                return

        self.file = open(self.file_path)

    def __del__(self):
        self.file.close()

    def log_line(self, line: str):
        self.file.write(f"{line}\n")
        if self.print_to_stdout:
            print(line)

    def log(self, something: str):
        self.file.write(something)
        if self.print_to_stdout:
            print(something, end="")

    def log_bar(self):
        bar = "============================================================\n"
        self.file.write(bar)
        if self.print_to_stdout:
            print(bar)
