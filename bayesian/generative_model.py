

# 生成模型
class GenerativeModel(object):

    def __init__(self, mu, sigma_s, s, sigma):
        self.mu = mu
        self.sigma_s = sigma_s
        self.s = s
        self.sigma = sigma

    # stimulus is magntitude variables
    def stimulus_distribution(self):


