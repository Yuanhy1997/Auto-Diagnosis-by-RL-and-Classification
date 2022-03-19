import numpy as np
from tqdm import tqdm
import pickle
import copy
from operator import itemgetter
import random
from scipy.stats import entropy

eps = 0.001

def sample_disease(sample_size, disease, prevalence = None):

    if prevalence is None:
        prevalence = np.ones(len(disease)) / len(disease)
    sampled_disease = np.random.choice(disease, sample_size, replace=True, p = prevalence)
    return sampled_disease

def sample_patients(sample_size, disease_dict, s2idx, prevalence = None):

    sampled_disease = sample_disease(sample_size, list(disease_dict.keys()))
    patients = {}
    s_count = 0
    init_count = 0
    s_index = np.random.poisson(2, len(sampled_disease)) + 1
    for i in tqdm(range(len(sampled_disease))):
        d = sampled_disease[i]
        num_s = np.min([np.random.poisson(8, 1)[0], len(disease_dict[d]['symptom'])])
        num_s = np.max([num_s, len(disease_dict[d]['symptom'])//2])
        s = np.random.choice(disease_dict[d]['symptom'], num_s, replace = False)
        s = list(itemgetter(*s)(s2idx))
        if len(disease_dict[d]['test']) > 1:
            t = list(itemgetter(*disease_dict[d]['test'])(s2idx))
        elif len(disease_dict[d]['test']) == 1:
            t = [s2idx[disease_dict[d]['test'][0]]]
        else:
            t = []
        patients[i] = {'d':d, 'self_report':s[:s_index[i]], 'acquired':s[s_index[i]:], 'test':t}
        patients[i]['all_sym'] = s+t
        s_count += len(s+t)
        init_count += len(patients[i]['self_report'])
    print('average symptom:', s_count/sample_size)
    print('average initial:', init_count/sample_size)
    return patients

class environment(object):
    def __init__(self, sample_size, args):
        
        self.sample_size = sample_size
        self.load_disease_dict()
        self.args = args
        self.patients = sample_patients(self.sample_size, self.disease_dict, self.s2idx)
        self.idx = 0
        self.indexes = np.arange(self.sample_size)
        self.diag_size = len(self.disease_dict)

        self.cost = np.concatenate((1 * np.ones(self.ss_size), 1 * np.ones(self.ts_size), 1 * np.ones(self.diag_size)))
        self.earn = np.concatenate((1 * np.ones(self.ss_size), 1 * np.ones(self.ts_size), 1 * np.ones(self.diag_size)))

    def idx_to_str(self, idx, idx_type = 'D'):
        
        if idx_type == 'D':
            return(list(self.d2idx.keys())[list(self.d2idx.values()).index(idx)])
        if idx_type == 'S':
            return(list(self.s2idx.keys())[list(self.s2idx.values()).index(idx)])

    def load_disease_dict(self):

        with open('../environment/medlineplus.pkl', 'rb') as f:
            buffer = pickle.load(f)
        self.disease_dict = buffer[0]
        self.s2idx = buffer[1]
        self.d2idx = buffer[2]
        self.ts_size = buffer[3][0] #test_symptom
        self.ss_size = buffer[3][1] #trivial_symptom symptom no need of test
        self.symptom_size = self.ts_size + self.ss_size

    def reset(self):

        self.idx = 0
        np.random.shuffle(self.indexes)
        
    def initialize_state(self, batch_size):

        self.batch_size = batch_size
        self.batch_index = self.indexes[self.idx : self.idx+batch_size]
        self.idx += batch_size
        self.disease = []
        self.pos_sym = []
        self.acquired_sym = []
        
        i = 0
        init_state = np.zeros((batch_size, self.symptom_size))
        self.all_state = np.zeros((batch_size, self.symptom_size))
        for item in self.batch_index:
            self.disease.append(self.d2idx[self.patients[item]['d']])
            self.all_state[i, self.patients[item]['all_sym']] = 1
            init_state[i, self.patients[item]['self_report']] = 1
            i += 1


        self.disease = np.array(self.disease)

        return init_state, self.disease

    def step(self, s, a_p, done, right_diagnosis, agent, ent_init, threshold, ent):

        s_ = copy.deepcopy(s)
        ent_ = copy.deepcopy(ent)
        s_[~done, a_p[~done]] =  self.all_state[~done, a_p[~done]] * 2 - 1

        a_d_, p_d_ = agent.choose_diagnosis(s_)

        ent_[~done] = entropy(p_d_[~done], axis = 1)
        ent_ratio = (ent-ent_) / ent_init

        diag = (ent_ < threshold[a_d_]) & (~done)
        right_diag = (a_d_ == np.array(self.disease)) & diag 

        reward_s = self.args.mu * self.reward_func(s, s_, diag, a_p) # add inquring reward
        reward_s[ent_ratio > 0] += (self.args.nu * ent_ratio[ent_ratio > 0]) # add entropy reward
        reward_s[diag] -= (1 * self.args.mu)# add diagnosing reward
        reward_s[right_diag] += (2 * self.args.mu)
        reward_s[done] = 0
        
        done += diag
        right_diagnosis += right_diag
        return s_, reward_s, done, right_diagnosis, diag, ent_, a_d_
    
    def reward_func(self, s, s_, diag, a_p):
        
        reward = -self.cost[a_p]
        reward += np.sum(np.abs(s-s_)*self.earn[:self.symptom_size], axis = 1) * 0.7
        reward += np.sum(((s_-s)>0)*self.earn[:self.symptom_size], axis = 1)

        return reward

if __name__ == '__main__':

    
    import json
    with open('../environment/medlineplus.json', 'r') as f:
        kb = json.load(f)
    d2idx = {}
    disease_dict = {}

    symptom_set = set()
    test_set = set()
    i = 0
    for disease in kb:
        d2idx[disease] = i
        disease_dict[disease] = {'test': [item[:10] for item in kb[disease]['test']], 'symptom': [item[:10] for item in kb[disease]['symptom']]}
        i += 1
        symptom_set.update([item[:10] for item in kb[disease]['symptom']])
        test_set.update([item[:10] for item in kb[disease]['test']])
    
    s2idx = {}
    i = 0
    for item in symptom_set:
        s2idx[item] = i
        i += 1
    for item in test_set:
        s2idx[item] = i
        i += 1
    print(len(symptom_set), len(test_set))
    print(len(symptom_set.union(test_set)))
    print(len(disease_dict))

    count_s = 0
    count_t = 0
    for disease in disease_dict:
        count_s += len(disease_dict[disease]['symptom'])
        count_t += len(disease_dict[disease]['test'])
    print(count_t/893, count_s/893)
    
    

    

