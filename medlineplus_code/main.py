from env import *
from agent import *
import copy
import torch
import pickle
import time
import numpy as np
from tqdm import tqdm
from scipy.stats import entropy
import argparse

def main():
    print("Initializing Environment and generating Patients....")
    env = environment(args.EPISODES, args)
    agent = Policy_Gradient_pair_model(state_size = env.symptom_size, disease_size = env.diag_size, LR = args.lr, Gamma = args.gamma)
    accuracy_list = []
    step_list = []
    reward_list = []
    positive_list = []
    threshold_list = []
    best = 0
    ave_step = 0

    if args.threshold_random_initial:
        threshold = np.random.rand(env.diag_size)
    else:
        threshold = args.threshold * np.ones(env.diag_size)

    best_a = 0
    best_p = 0
    for epoch in range(args.EPOCHS):
        env = environment(args.EPISODES, args)
        for episode in range(args.EPISODES//args.GAMES):
            states = []
            action_m = []
            rewards_s = []
            action_s = []
            true_d = []

            s, true_disease = env.initialize_state(args.GAMES)
            s_init = copy.deepcopy(s)
            s_final = copy.deepcopy(s)

            a_d, p_d = agent.choose_diagnosis(s)
            init_ent = entropy(p_d, axis = 1)
            
            done = (init_ent < threshold[a_d])
            right_diag = (a_d == env.disease) & done

            diag_ent = np.zeros(args.GAMES)
            finl_diag = np.zeros(args.GAMES).astype(np.int) - 1
            diag_ent[right_diag] = init_ent[right_diag]
            ent = init_ent

            for i in range(args.MAXSTEP):

                a_s = agent.choose_action_s(s)
                s_, r_s, done, right_diag, final_idx, ent_, a_d_ = env.step(s, a_s, done, right_diag, agent, init_ent, threshold, ent)

                s_final[final_idx] = s_[final_idx]
                diag_ent[right_diag] = ent_[right_diag]
                finl_diag[right_diag] = a_d_[right_diag]

                if i == args.MAXSTEP - 1:
                    r_s[done==False] -= 1

                states.append(s)
                rewards_s.append(r_s)
                action_s.append(a_s)
                true_d.append(true_disease)
                
                s = s_
                ent = ent_
                
                if all(done):
                    break

            diag = np.sum(done) + 1e-5
            s_final[~done] = s_[~done]
                
            ave_step, ave_reward_s = agent.create_batch(states, rewards_s, action_s, true_d)
            a_d, p_d = agent.choose_diagnosis(s)

            t_d = (a_d == env.disease) & (~done)
            diag_ent[t_d] = entropy(p_d[t_d], axis = 1)
            finl_diag[t_d] = a_d[t_d]

            for idx, item in enumerate(finl_diag):
                if item >= 0 and abs(threshold[item] - diag_ent[idx]) > 0.01:
                    threshold[item] = (args.lamb * threshold[item] + (1-args.lamb) * diag_ent[idx])

            agent.update_param_rl()
            agent.update_param_c()

            ave_pos = (np.sum(s_final == 1) - np.sum(s_init == 1)) / ave_step
            step_list.append(ave_step/args.GAMES)
            reward_list.append(ave_reward_s)
            positive_list.append(ave_pos)
            threshold_list.append(threshold)
            accuracy_list.append((sum(right_diag)+sum(a_d[done == False] == env.disease[done == False]))/args.GAMES)
            best_a = np.max([best_a, accuracy_list[-1]])

            print("==Epoch:", epoch+1, '\tAve. Accu:', accuracy_list[-1], '\tBest Accu:', best_a, '\tAve. Pos:', ave_pos)
            print('Threshold:', threshold[:5], '\tAve. Step:', ave_step/args.GAMES, '\tAve. Reward Sym.:', ave_reward_s, '\n')
        
        agent.save_model(args)
        info =  str(args.threshold) + '_' + str(args.mu) + '_' + str(args.nu) + '_' + str(args.trail)
        with open('./stats_logging_'+info+'.pkl','wb') as f:
            pickle.dump([accuracy_list, step_list, reward_list, positive_list, threshold_list], f)

def test():
    print("Initializing Environment and generating Patients....")
    env = environment(args.EPISODES, args)
    agent = Policy_Gradient_pair_model(state_size = env.symptom_size, disease_size = env.diag_size, LR = 1e-4, Gamma = 0.99)
    agent.load_model(args)

    info = str(args.threshold) + '_' + str(args.mu) + '_' + str(args.nu) + '_' + str(args.trail)
    with open('./stats_logging_'+info+'.pkl','rb') as f:
        buffer = pickle.load(f)
    threshold = buffer[-1][-1]

    steps_on_ave = 0
    pos_on_ave = 0
    accu_on_ave = 0

    for episode in range(args.EPISODES//args.GAMES):
        states = []
        action_m = []
        rewards_s = []
        action_s = []
        true_d = []

        s, true_disease = env.initialize_state(args.GAMES)
        s_init = copy.deepcopy(s)
        s_final = copy.deepcopy(s)

        a_d, p_d = agent.choose_diagnosis(s)
        init_ent = entropy(p_d, axis = 1)
        
        done = (init_ent < threshold[a_d])
        right_diag = (a_d == env.disease) & done

        diag_ent = np.zeros(args.GAMES)
        diag_ent[right_diag] = init_ent[right_diag]
        ent = init_ent

        for i in range(args.MAXSTEP):

            a_s = agent.choose_action_s(s)
            s_, r_s, done, right_diag, final_idx, ent_, a_d_ = env.step(s, a_s, done, right_diag, agent, init_ent, threshold, ent)

            s_final[final_idx] = s_[final_idx]
            diag_ent[right_diag] = ent_[right_diag]

            if i == args.MAXSTEP - 1:
                r_s[done==False] -= 1

            states.append(s)
            rewards_s.append(r_s)
            action_s.append(a_s)
            true_d.append(true_disease)
            
            s = s_
            ent = ent_
            
            if all(done):
                break

        diag = np.sum(done) + 1e-5
        s_final[~done] = s_[~done]
            
        all_step, ave_reward_s = agent.create_batch(states, rewards_s, action_s, true_d)
        ave_step = all_step / args.GAMES
        a_d, p_d = agent.choose_diagnosis(s)
        finl_ent = entropy(p_d, axis = 1)
        t_d = (a_d == env.disease) & (~done)
        diag_ent[t_d] = finl_ent[t_d]
        
        accurate = (sum(right_diag) + sum(t_d)) / args.GAMES
        ave_pos = (np.sum(s_final == 1) - np.sum(s_init == 1)) / all_step

        steps_on_ave = episode / (episode + 1) * steps_on_ave + 1 / (episode + 1) * ave_step
        pos_on_ave = episode / (episode + 1) * pos_on_ave + 1 / (episode + 1) * ave_pos
        accu_on_ave = episode / (episode + 1) * accu_on_ave + 1 / (episode + 1) * accurate
        
        print(steps_on_ave, pos_on_ave, accu_on_ave)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process Settings')
    parser.add_argument('-seed', type=int, default = 514,
                        help='set a random seed')
    parser.add_argument('-threshold', type=float, default = 1,
                        help='set the initial threshold')
    parser.add_argument('-threshold_random_initial', action="store_true",
                        help='randomly initialize threshold')
    parser.add_argument('-EPISODES', type=int, default = 100000,
                        help='episodes of MAD rounds per epoch')
    parser.add_argument('-GAMES', type=int, default = 200,
                        help='games for each time onpolicy sample collection')
    parser.add_argument('-EPOCHS', type=int, default = 200,
                        help='training epochs')
    parser.add_argument('-MAXSTEP', type=int, default = 15,
                        help='max inquiring turns of each MAD round')
    parser.add_argument('-nu', type=float, default = 2.5,
                        help='nu')
    parser.add_argument('-mu', type=float, default = 1,
                        help='mu')
    parser.add_argument('-lr', type=float, default = 1e-4,
                        help='learning rate')
    parser.add_argument('-gamma', type=float, default = 0.99,
                        help='reward discount rate')
    parser.add_argument('-train', action="store_true",
                        help='whether test on the exsit result model or train a new model')
    parser.add_argument('-trail', type=int, default = 1,
                        help='the trail number')
    parser.add_argument('-lamb', type=float, default = 0.99,
                        help='polyak factor for threshold adjusting')
    args = parser.parse_args()

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if args.train:
        main()
    else:
        test()
