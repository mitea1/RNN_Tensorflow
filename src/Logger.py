class Logger:

    def __init__(self):
        self.logs = []

    def add_log(self, log):
        self.logs.append(log)

    def get_log(self, index):
        return self.logs[index]