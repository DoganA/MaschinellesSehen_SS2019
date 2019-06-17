import numpy as np
import matplotlib.pyplot as plt


class Line:
    """helper class"""

    def __init__(self, point1, point2):
        # y = mx + b
        self.m = (point2[1] - point1[1])/(point2[0] - point1[0])
        self.b = point1[1] - self.m*point1[0]
        self.point1 = point1
        self.point2 = point2

    def plot(self, x_range=(0,1)):
        """
        Plots the line in the x range. Default range ist between 0 and 1.
        :param x_range: tuple
        :return: distance
        """
        plt.plot(x_range, np.array([self.m*x_range[0] + self.b, self.m*x_range[1] + self.b]))

    def distance(self, p):
        """
        Compute the distance of a point p to a the line
        :param p: Point
        :return: distance
        """
        return np.fabs(self.m * p[0] - p[1] + self.b) / np.sqrt(1 + self.m * self.m)

    def plot_with_threshold(self, threshold, x_range=(0,1)):
        """
        Plots the line in the x range with threshold made visible. Default range ist between 0 and 1.
        :param threshold: the threshold
        :param x_range: tuple
        :return: distance
        """
        self.plot(x_range)
        plt.plot(x_range, np.array([self.m*x_range[0] + self.b+threshold, self.m*x_range[1] + self.b+threshold]),"--r")
        plt.plot(x_range, np.array([self.m*x_range[0] + self.b-threshold, self.m*x_range[1] + self.b-threshold]),"--r")

    def __repr__(self):
        return f"Line({self.point1},{self.point2})"


def RansacPointGenerator(numpointsInlier, numpointsOutlier):
        pure_x = np.linspace(0, 1, numpointsInlier)
        pure_y = np.linspace(0, 1, numpointsInlier)
        noise_x = np.random.normal(0, 0.025, numpointsInlier)
        noise_y = np.random.normal(0, 0.025, numpointsInlier)

        outlier_x = np.random.random_sample((numpointsOutlier,))
        outlier_y = np.random.random_sample((numpointsOutlier,))

        points_x = pure_x + noise_x
        points_y = pure_y + noise_y
        points_x = np.append(points_x, outlier_x)
        points_y = np.append(points_y, outlier_y)

        return np.array([points_x, points_y])


if __name__ == "__main__":
    # --------------------------------------------------------------------------
    # DATA INITALISATION
    # --------------------------------------------------------------------------
    # generate random x and y data with 100 inliers and 10 outliers
    x,y = RansacPointGenerator(100,50)

    # plot the datapoints
    plt.plot(x,y, ".")
    plt.show()

    # Run RANSAC M times with threshold:
    M = 10
    threshold = 0.1

    # init the best model with None (because we have no model found so far)
    bestModel = None

    inline_peak = 0

    # --------------------------------------------------------------------------
    # THE ALGORITHM
    # --------------------------------------------------------------------------
    # in every step:
    for i in range(M):

        # pick a random sample (two Points). Hint: np.random.randint(min, max, cout)
        zahlen = np.arange(0, len(x),1)

        sample = np.random.choice(zahlen, 2)
        random1 = sample[0]
        random2 = sample[1]

        random = np.random.randint(0, len(x), 1)[0]
        p1 = (x[random1], y[random1])

        p2 = (x[random2], y[random2])

        # generate a model using this sample. Hint: use the Line class.
        line = Line(p1, p2)

        # plot the model with threshold and the datapoints. Hint: the Line class has a plot_with_threshold function. (optional)
        #line.plot_with_threshold(threshold)
        line.plot()
        plt.plot(x, y, ".")
        plt.show()

        # count the inliers
        count=0
        for p in zip(x,y):
            distance = line.distance(p)
            if distance <= threshold:
                count +=1

        # save model, if it is better (has more inliers) than all previous models
        if count > inline_peak:
            inline_peak = count
            bestModel = line

    # plot Data with found model:
    bestModel.plot(9)
    plt.show()
    plt.
