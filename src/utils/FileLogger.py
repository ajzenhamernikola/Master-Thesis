import os


class FileLogger(object):
    def __init__(self, file_root, file_name, print_to_stdout=True):
        self.file_path = os.path.abspath(os.path.join(file_root, f"report_{file_name}.txt"))
        self.file = None
        self.print_to_stdout = print_to_stdout
        self.overwrite = True
        self.buffer = []

        # Checks
        if os.path.exists(self.file_path):
            print(f"The file {self.file_path} already exists. Overwrite? [y/n]")
            self.overwrite = input().lower() == "y"

    def __del__(self):
        with open(self.file_path, "w" if self.overwrite else "a") as file:
            file.write(''.join(self.buffer))

    def log_line(self, line: str):
        self.buffer.append(f"{line}\n")
        if self.print_to_stdout:
            print(line)

    def log(self, something: str):
        self.buffer.append(something)
        if self.print_to_stdout:
            print(something, end="")

    def log_bar(self):
        bar = "============================================================"
        self.buffer.append(f"{bar}\n")
        if self.print_to_stdout:
            print(bar)
