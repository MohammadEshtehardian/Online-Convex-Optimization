import numpy as np

class BanditSolver:

    # constructor
    def __init__(self, arms):
        self.arms = arms

    # exp3 solver
    def exp3(self, epsilon, iterations):
        n = len(self.arms)
        # initialize probabilities
        x = 1/n * np.ones((n, ))
        loss_history = np.zeros((iterations, ))
        for t in range(iterations):
            # choose a random arm
            i = int(np.random.choice(np.linspace(0, n-1, n), p=x))
            # calculate the loss
            loss = self.arms[i].loss() / x[i]
            loss_history[t] = (loss_history[t-1]*t + loss)/(t+1)
            # update probabilities
            x[i] = x[i] * np.exp(-epsilon*loss)
            x = x / np.sum(x)
        return x, loss_history
