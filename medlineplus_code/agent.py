import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.nn as nn
import numpy as np
import tqdm
import pickle

CUDA_device = 0
eps = 1e-12

class sym_acquire_func(nn.Module):

    def __init__(self, state_size, action_size):
        super(sym_acquire_func, self).__init__()
        self.fc1 = nn.Linear(state_size, 1024*5)
        self.fc2 = nn.Linear(1024*5, 2048*5)
        self.fc3 = nn.Linear(2048*5, 1024*5)
        self.out = nn.Linear(1024*5, action_size)

    def forward(self,x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = F.gelu(x)
        x = self.fc3(x)
        x = F.relu(x)
        # action_prob = torch.exp(F.log_softmax(self.out(self.batch_norm(x)), dim = 1))
        action_prob = torch.exp(F.log_softmax(self.out(x), dim = 1))

        return action_prob

class diagnosis_func(nn.Module):

    def __init__(self, state_size, disease_size):
        super(diagnosis_func, self).__init__()
        self.fc1 = nn.Linear(state_size, 1024*5)
        self.fc2 = nn.Linear(1024*5, 1024*5)
        self.out = nn.Linear(1024*5, disease_size)

    def forward(self,x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = F.relu(x)
        output = self.out(x)
        return output


class Policy_Gradient_pair_model(object):
    def __init__(self, state_size, disease_size, LR = 1e-4, Gamma = 0.99, Eta = 0.01):

        self.policy = sym_acquire_func(state_size, state_size)
        self.classifier = diagnosis_func(state_size, disease_size)
        self.lr = LR
        self.policy.cuda(CUDA_device)
        self.classifier.cuda(CUDA_device)
        self.optimizer_p = torch.optim.Adam(self.policy.parameters(), lr=LR/5)
        self.optimizer_c = torch.optim.Adam(self.classifier.parameters(), lr=LR)
        self.counter = 1
        self.cross_entropy = nn.CrossEntropyLoss()
        self.gamma = Gamma
        self.eta = Eta

    def create_batch(self, states, rewards_s, action_s, true_d):
        
        cumulate_R_s = []
        R_s = 0
        for r_s in rewards_s[::-1]:
            R_s = r_s + self.gamma * R_s
            cumulate_R_s.insert(0, R_s)
        
        rewards_s = np.array(rewards_s)
        ave_rewards_s = np.mean(np.sum(rewards_s, axis = 0))

        cumulate_R_s = np.array(cumulate_R_s).T
        states = np.array(states).swapaxes(0, 1)
        action_s = np.array(action_s).T
        true_d = np.array(true_d).T

        valid_sample = (cumulate_R_s != 0)

        self.batch_rewards_s = torch.from_numpy(cumulate_R_s[valid_sample]).float()
        self.batch_states = torch.from_numpy(states[valid_sample]).float()
        self.batch_action_s = torch.from_numpy(action_s[valid_sample])
        self.batch_true_d = torch.from_numpy(true_d[valid_sample])

        return len(self.batch_rewards_s), ave_rewards_s
     
    def choose_action_s(self, state):

        self.policy.eval()
        state = torch.from_numpy(state).float()
        probs = self.policy.forward(state.cuda(CUDA_device))
        m = Categorical(probs)
        action = m.sample().detach().cpu().squeeze().numpy()

        return action
    
    def choose_diagnosis(self, state):

        self.classifier.eval()
        state = torch.from_numpy(state).float()
        output = self.classifier.forward(state.cuda(CUDA_device)).detach().cpu().squeeze()

        return torch.max(output, dim = 1)[1].numpy(), torch.softmax(output, dim = 1).numpy()

    def update_param_rl(self):  

        self.policy.train() 
        self.optimizer_p.zero_grad()
        state_tensor = self.batch_states.cuda(CUDA_device)
        reward_tensor = self.batch_rewards_s.cuda(CUDA_device)
        action_s_tensor = self.batch_action_s.cuda(CUDA_device)
        prob_tensor = self.policy.forward(state_tensor)
        #Policy Loss
        m = Categorical(prob_tensor)
        log_prob_tensor = m.log_prob(action_s_tensor)
        policy_loss = - (log_prob_tensor * (reward_tensor)).mean()
        #entropy Loss
        entropy_loss = - torch.max(torch.tensor([self.eta-self.counter*0.00001, 0])) * m.entropy().mean()
        loss = policy_loss + entropy_loss
        loss.backward()
        self.optimizer_p.step()

        self.counter += 1

    def update_param_c(self):

        self.classifier.train()
        self.optimizer_c.zero_grad()
        state_tensor = self.batch_states.cuda(CUDA_device)
        label_tensor = self.batch_true_d.cuda(CUDA_device)
        output_tensor = self.classifier.forward(state_tensor)
        loss = self.cross_entropy(output_tensor, label_tensor)
        loss.backward()
        self.optimizer_c.step()
  
    def save_model(self, args):
        info = str(args.threshold) + '_' + str(args.mu) + '_' + str(args.nu)+ '_' + str(args.trail)
        torch.save(self.policy.state_dict(), './model_saved/policy_'+info+'.pth')
        torch.save(self.classifier.state_dict(), './model_saved/classifier_'+info+'.pth')
    
    def load_model(self, args):
        info = str(args.threshold) + '_' + str(args.mu) + '_' + str(args.nu) + '_' + str(args.trail)
        self.policy.load_state_dict(torch.load('./model_saved/policy_'+info+'.pth', map_location='cuda:0'))
        self.classifier.load_state_dict(torch.load('./model_saved/classifier_'+info+'.pth', map_location='cuda:0'))
