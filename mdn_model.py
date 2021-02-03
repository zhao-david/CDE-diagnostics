import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb


class MDNPerceptron(nn.Module):
    def __init__(self, n_input, n_hidden, n_gaussians):
        nn.Module.__init__(self)
        self.n_gaussians = n_gaussians
        self.l1 = nn.Linear(n_input, n_hidden)

        self.z_pi = nn.Linear(n_hidden, n_gaussians)
        #self.z_mu = nn.Linear(n_hidden, n_gaussians)
        #self.z_sigma = nn.Linear(n_hidden, n_gaussians)
        self.z_mu = nn.Linear(n_hidden, 2*n_gaussians)
        self.z_sigma = nn.Linear(n_hidden, 2*n_gaussians)

    """ Returns parameters for a mixture of gaussians given x
    mu - vector of means of the gaussians
    sigma - vector of the standard deviation of the gaussians
    pi - probability distribution over the gaussians
    """
    def forward(self, x):
        hidden = torch.tanh(self.l1(x))

        pi = F.softmax(self.z_pi(hidden), 1)
        mu = self.z_mu(hidden)
        sigma = torch.exp(self.z_sigma(hidden))

        return pi, mu, sigma

    """Makes a random draw from a randomly selected
    mixture based upon the probabilities in Pi
    """
    def sample(self, pi, mu, sigma):
        mixture = torch.normal(mu, sigma)
        k = torch.multinomial(pi, 1, replacement=True).squeeze()
        result = mixture[range(k.size(0)), k]
        return result

    """Computes the log probability of the datapoint being
    drawn from all the gaussians parametized by the network.
    Gaussians are weighted according to the pi parameter 
    """
    def loss_fn(self, y, pi, mu, sigma):
        #mixture = torch.distributions.normal.Normal(mu, sigma)
        probs = []
        for i in range(self.n_gaussians):
            z = torch.zeros(size=[mu.size(0)])
            cov = torch.stack([
                sigma[:, 2*i], z,
                z, sigma[:, 2*i+1]
            ], dim=-1).view(-1, 2, 2)
            mixture = torch.distributions.multivariate_normal.MultivariateNormal(mu[:, 2*i:2*i+2], cov)
            log_prob = mixture.log_prob(y)
            prob = torch.exp(log_prob)
            probs.append(prob)
        probs = torch.stack(probs).T
        #weighted_prob = prob * pi
        weighted_prob = probs * pi
        sum = torch.sum(weighted_prob, dim=1)
        log_prob_loss = -torch.log(sum)
        ave_log_prob_loss = torch.mean(log_prob_loss)
        return ave_log_prob_loss