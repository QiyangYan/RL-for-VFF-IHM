import numpy as np
np.random.seed(0)


class RandomisationModule:
    def __init__(self):
        self.randomisation_list = {
            # lower bound, upper bound
            "damping": [4.7114, 5.3059],  # without object [4.7114, 5.3059]
            "armature": [0.37645,  0.4241],
            # "frictionloss": [],
            "kp": [12.7, 13],  # without object [12.7, 13]
            "floor_friction": [0.01, 1]
        }
        self.gaussian_noise_list = {
            # mean, std
            "object_position": [0.001, 0.0005],
            "object_orientation": [],
            "joint_position": [0, 0.007],
            "hand_base_marker_position": [],
        }
        self.gaussian_noise_cov_matrix_list = {
            "object_position": [],
            "object_orientation": [],
            "dynamixel_position": []
        }

    def uniform_randomise(self, term):
        """
        Generates values from a log-uniform distribution.

        :param term: randomisation target
        :return: uniform([log_low, log_high])
        """
        assert self.randomisation_list[term] != [], f"{term} in randomisation list is empty"
        return np.random.uniform(*self.randomisation_list[term])

    def log_uniform_randomise(self, term):
        """
        Generates values from a log-uniform distribution.

        :param term: randomisation target
        :return: loguniform([log_low, log_high])
        """
        assert self.randomisation_list[term] != [], f"{term} in randomisation list is empty"
        # print(self.randomisation_list[term])
        log_low = np.log(self.randomisation_list[term][0])
        log_high = np.log(self.randomisation_list[term][1])
        # print(log_low, log_high)
        log_uniform = np.random.uniform(log_low, log_high)
        return np.exp(log_uniform)

    def generate_gaussian_noise(self, term, size, correlated):
        """
        Generates uncorrelated + correlated (optional) Gaussian (normal) noise.

        :param term: name of the randomisation term
        :param size: The shape of the output array. Default is 1, generating a single value.
                     Can be an integer or a tuple for generating arrays of noise.
        :param correlated: generate correlated or only uncorrelated value
        :return: Gaussian noise with the specified parameters.
        """

        mean = self.gaussian_noise_list[term][0]
        std = self.gaussian_noise_list[term][1]
        uncorrelated_noise = np.random.normal(loc=mean, scale=std, size=size)

        correlated_noise = 0
        if correlated:
            correlated_noise = np.random.multivariate_normal(mean, self.gaussian_noise_cov_matrix_list[term], size)

        return uncorrelated_noise + correlated_noise


if __name__ == "__main__":
    randomisation = RandomisationModule()
    print(randomisation.uniform_randomise(term='damping'))



