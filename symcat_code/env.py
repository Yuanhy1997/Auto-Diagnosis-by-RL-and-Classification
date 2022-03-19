import numpy as np
from tqdm import tqdm
import pickle
import copy
from operator import itemgetter
import random
from scipy.stats import entropy

eps = 0.001
gender2idx = {'Male':0,'Female':1}
age2idx = {"< 1": 0, "1-4": 1, "5-14": 2, "15-29": 3, "30-44": 4, "45-59": 5, "60-74": 6, "75+": 7}

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
    for i in tqdm(range(len(sampled_disease))):
        d = sampled_disease[i]
        while(True):
            syms = (np.random.uniform(size = len(disease_dict[d]['prevalence'])) - np.array(disease_dict[d]['prevalence']) ) < 0
            if np.sum(syms)>=1:
                break
        syms = np.array(disease_dict[d]['symptom'])[syms]
        if len(syms) > 1:
            s = list(itemgetter(*syms)(s2idx))
        else:
            s = [s2idx[syms[0]]]
        age = np.random.choice(disease_dict[d]['age'], 1, p = disease_dict[d]['age_p'])[0]
        gender = np.random.choice(disease_dict[d]['gender'], 1, p = disease_dict[d]['gender_p'])[0]
        patients[i] = {'d':d, 'self_report':s[:1], 'acquired':s[1:], 'test':[], 'age':age2idx[age], 'gender':gender2idx[gender]}
        patients[i]['all_sym'] = s
        s_count += len(s)
        init_count += len(patients[i]['self_report'])
    print('average symptom:', s_count/sample_size)
    print('average initial:', init_count/sample_size)
    return patients

class environment(object):
    def __init__(self, sample_size, args):
        self.args = args
        self.dataset = self.args.dataset
        self.context_size = len(age2idx) + len(gender2idx)
        self.sample_size = sample_size
        self.load_disease_dict()
        self.patients = sample_patients(self.sample_size, self.disease_dict, self.s2idx)
        self.idx = 0
        self.indexes = np.arange(self.sample_size)
        self.diag_size = len(self.disease_dict)

        self.cost = np.concatenate((1 * np.ones(self.ss_size), 1 * np.ones(self.ts_size)))
        self.earn = np.concatenate((1 * np.ones(self.ss_size), 1 * np.ones(self.ts_size)))

    def load_disease_dict(self):

        with open('../environment/symcat_data_'+self.dataset+'_withcontext.pkl', 'rb') as f:
            buffer = pickle.load(f)
        self.disease_dict = buffer[0]
        self.s2idx = buffer[1]
        self.d2idx = buffer[2]
        self.ts_size = buffer[3][0] #test_symptom
        self.ss_size = buffer[3][1] #trivial_symptom symptom no need of test
        self.symptom_size = self.ts_size + self.ss_size
        print(self.symptom_size)

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
        init_state = np.zeros((batch_size, self.symptom_size+self.context_size))
        self.all_state = np.zeros((batch_size, self.symptom_size))
        for item in self.batch_index:
            self.disease.append(self.d2idx[self.patients[item]['d']])
            self.all_state[i, self.patients[item]['all_sym']] = 1
            init_state[i, self.patients[item]['self_report']] = 1
            init_state[i, self.patients[item]['age']+self.symptom_size] = 1
            init_state[i, self.patients[item]['gender']+self.symptom_size+len(age2idx)] = 1
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

        reward_s = self.args.mu * self.reward_func(s[:,:self.symptom_size], s_[:,:self.symptom_size], diag, a_p) 
        reward_s[ent_ratio > 0] += (self.args.nu * ent_ratio[ent_ratio > 0])
        reward_s[diag] -= (self.args.mu * 1)
        reward_s[right_diag] += (self.args.mu * 2)
        reward_s[done] = 0
        
        done += diag
        right_diagnosis += right_diag
        
        return s_, reward_s, done, right_diagnosis, diag, ent_, a_d_
    
    def reward_func(self, s, s_, diag, a_p):
        
        reward = -self.cost[a_p]
        reward += np.sum(np.abs(s-s_)*self.cost, axis = 1) * 0.7
        reward += np.sum(((s_-s)>0)*self.earn, axis = 1)

        return reward

if __name__ == '__main__':
    np.random.seed(1000)
    random.seed(1000)
    env = environment(100)