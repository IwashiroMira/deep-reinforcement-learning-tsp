import matplotlib.pyplot as plt


class Plot():
    def __init__(self, data):
        self.data = data

    def plot(self):
        plt.plot(self.data)
        plt.title('Data Plot')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.show()