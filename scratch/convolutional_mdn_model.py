import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvMDNPerceptron(nn.Module):
    def __init__(self, n_hidden, n_gaussians):
        nn.Module.__init__(self)
        self.n_gaussians = n_gaussians
        
        self.conv1 = nn.Conv2d(1,6,3,1)
        self.conv2 = nn.Conv2d(6,12,3,1)
        self.fc1 = nn.Linear(768, 128)
        self.fc2 = nn.Linear(128, n_hidden)

        self.z_pi = nn.Linear(n_hidden, n_gaussians)
        self.z_mu = nn.Linear(n_hidden, n_gaussians)
        self.z_sigma = nn.Linear(n_hidden, n_gaussians)

    """ Returns parameters for a mixture of gaussians given x
    mu - vector of means of the gaussians
    sigma - vector representing the diagonals of the covariances of the gaussians
    pi - probability distribution over the gaussians
    """
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        hidden = self.fc2(x)
        
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
        mixture = torch.distributions.normal.Normal(mu, sigma)
        log_prob = mixture.log_prob(y.reshape(-1,1))
        prob = torch.exp(log_prob)
        weighted_prob = prob * pi
        sum = torch.sum(weighted_prob, dim=1)
        log_prob_loss = -torch.log(sum)
        ave_log_prob_loss = torch.mean(log_prob_loss)
        return ave_log_prob_loss