import numpy as np


class NBC:
    EPS = 0.000000000001
    alpha = 1  # smoothing parameter for Laplace smoothing

    def __init__(self, feature_types, num_classes):
        self.r_features_indices = [i for i in range(len(feature_types))
                                   if feature_types[i] == 'r']
        self.b_features_indices = [i for i in range(len(feature_types))
                                   if feature_types[i] == 'b']
        self.rD = feature_types.count('r')  # num real features
        self.bD = feature_types.count('b')  # num binary features
        self.C = num_classes                # number of classes
        self.rparams_mean = np.ones((self.rD, self.C))
        self.rparams_var = np.ones((self.rD, self.C))
        self.bparams = np.ones((self.bD, self.C))
        self.log_pi_c = np.ones(self.C)     # log p(y = c | params)

    # determine optimal parameters, given some data
    def fit(self, X, y):
        (N, D) = X.shape
        assert D == self.rD + self.bD
        X_r = X[:, self.r_features_indices]
        X_b = X[:, self.b_features_indices]

        # this creates a sort of mask that can be multiplied with
        # X (reshaped into (N, D, C) to split features between the classes
        # they belong to)

        M = np.eye(N, M=self.C)[y]

        N_c = np.sum(M, axis=0)  # number of data points per class
        pi_c = N_c / N
        self.log_pi_c = np.log(np.where(pi_c < NBC.EPS, NBC.EPS, pi_c))

        M = M[:, np.newaxis, :]

        # real parameters estimation
        # np.where is used to prevent division by 0 & to ensure an unbiased
        # variance estimator
        self.rparams_mean = np.sum(X_r[:, :, np.newaxis] * M, axis=0) \
            / np.where(N_c == 0, 1, N_c)

        r_var =  \
            np.sum(((X_r[:, :, np.newaxis] - self.rparams_mean) * M) ** 2,
                   axis=0) \
            / np.where(N_c <= 1, 1, N_c - 1)
        # make sure variances are not too small
        self.rparams_var = np.where(r_var < NBC.EPS, NBC.EPS, r_var)

        # binary parameters estimation with smoothing
        self.bparams = \
            (np.sum(X_b[:, :, np.newaxis] * M, axis=0) + NBC.alpha) \
            / (N_c + NBC.alpha * self.bD)

    def predict(self, X):
        (N, D) = X.shape
        assert D == self.rD + self.bD
        X_r = X[:, self.r_features_indices]
        X_b = X[:, self.b_features_indices]

        # Handle real-valued features
        X_r = X_r[:, :, np.newaxis]

        LL_r = -(X_r - self.rparams_mean) ** 2 / self.rparams_var \
            - np.log(self.rparams_var)

        # Handle binary-valued features
        X_b = X_b[:, :, np.newaxis]

        LL_b = np.log(X_b * self.bparams) + \
            np.log((1 - X_b) * (1 - self.bparams))

        # Marginalize the likelihood over the features
        LL_r = np.sum(LL_r, axis=1)
        LL_b = np.sum(LL_b, axis=1)

        LL = LL_r + LL_b + self.log_pi_c  # shape = (N, C)

        # Marginalize the likelihood over the data points and over the
        # features, then find the argmax of the resulting class
        # likelihood marginals
        y_pred = np.argmax(LL, axis=1)  # shape = N
        return y_pred.T
