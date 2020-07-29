class IntProgress(object):
    def __init__(self, min: int, max: int):
        self.min = min
        self.max = max
        self.percentages = [int(min + ((max - min) / 100) * p) for p in range(10, 100, 10)]
        self.percentages.append(max)
        self.count: int = None
        self.prev_percentage: int = None
        self.reset()

    def reset(self):
        self.count = self.min
        self.prev_percentage = self.percentages[0]

    def step(self, count: int, end=" "):
        if self.count == self.min:
            print("\t", end="")
        self.count += count

        for p in self.percentages:
            if self.count < p:
                break

            if self.count > self.prev_percentage:
                i = self.percentages.index(self.prev_percentage)
                print(f"{10 * (i + 1)}%", end=end, flush=True)
                if i + 1 < len(self.percentages):
                    self.prev_percentage = self.percentages[i + 1]
                else:
                    self.prev_percentage = self.max
                break
