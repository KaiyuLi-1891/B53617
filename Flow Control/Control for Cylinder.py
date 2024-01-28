import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.optim import Adam
import time
from tqdm import tqdm
from torch import distributions


class get_actor(nn.Module):
    def __init__(self, n_states, n_hiddens, n_actions, Lim):
        super(get_actor, self).__init__()
        self.lim = Lim
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_hiddens)
        self.fc_mu = nn.Linear(n_hiddens, n_actions)
        self.fc_sig = nn.Linear(n_hiddens, n_actions)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)  # tanh()
        x = self.fc2(x)
        x = F.relu(x)
        mu = self.fc_mu(x)
        mu = self.lim * torch.tanh(mu)
        sig = self.fc_sig(x)
        sig = F.softplus(sig)
        return mu, sig

    def save_checkpoint(self, ckpt_file):
        torch.save(self.state_dict(), ckpt_file)

    def load_checkpoint(self, ckpt_file):
        self.load_state_dict(torch.load(ckpt_file))


class get_critic(nn.Module):
    def __init__(self, n_states, n_hiddens):
        super(get_critic, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_hiddens)
        self.fc3 = nn.Linear(n_hiddens, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        return x

    def save_checkpoint(self, ckpt_file):
        torch.save(self.state_dict(), ckpt_file)

    def load_checkpoint(self, ckpt_file):
        self.load_state_dict(torch.load(ckpt_file))


class PPO:
    def __init__(self, Env, n_states, n_hiddens, n_actions, ac_lr, cr_lr, Lambda, Gamma, Lim, Epochs, clip_eps):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = Env

        self.actor = get_actor(n_states, n_hiddens, n_actions, Lim).to(self.device)
        self.critic = get_critic(n_states, n_hiddens).to(self.device)
        self.ac_lr = ac_lr
        self.cr_lr = cr_lr
        self.ac_opti = Adam(self.actor.parameters(), lr=self.ac_lr)
        self.cr_opti = Adam(self.critic.parameters(), lr=self.cr_lr)

        self.lmbda = Lambda
        self.gamma = Gamma
        self.lim = Lim
        self.epochs = Epochs
        self.clip_epsilon = clip_eps
        self.max_grad_norm = 0.5
        self.batch = 20
        self.list_reward = []
        self.list_drag = []
        self.list_list = []

    def save_model(self, ac_file, cr_file):
        self.actor.save_checkpoint(ac_file)
        self.critic.save_checkpoint(cr_file)

    def get_action(self, state):
        s = torch.tensor(state, dtype=torch.float).to(self.device)
        mu, sig = self.actor(s)
        dist = distributions.Normal(mu, sig)
        a = dist.sample()
        a = torch.clamp(a, -self.lim, self.lim)
        return a.item()

    def update(self, state, action, reward, state_):
        s = torch.tensor([state], dtype=torch.float).to(self.device)
        a = torch.tensor([action], dtype=torch.float).view(-1, 1).to(self.device)
        r = torch.tensor([reward], dtype=torch.float).view(-1, 1).to(self.device)
        s_ = torch.tensor([state_], dtype=torch.float).to(self.device)

        target_v_ = self.critic(s_)
        target_td = r + self.gamma * target_v_
        td = self.critic(s)
        td_er = (td - target_td).cpu().detach().numpy()
        gae_list = []
        gae = 0
        for i in td_er[::-1]:
            # print(i,'/')
            gae = self.gamma * self.lmbda * gae + i
            gae_list.append(gae)
        gae_list.reverse()
        gae_list = np.array(gae_list)
        gae_ = torch.tensor(gae_list, dtype=torch.float).to(self.device)

        mu, sig = self.actor(s)
        dist = distributions.Normal(mu.detach(), sig.detach())  ##Beta
        old_logprob = dist.log_prob(a)

        for _ in range(self.epochs):
            mu, sig = self.actor(s)
            dist = distributions.Normal(mu, sig)
            _logprob = dist.log_prob(a)
            ratio = torch.exp(_logprob - old_logprob)

            L1 = ratio * gae_
            L2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * gae_

            ac_loss = -torch.mean(torch.min(L1, L2))
            cr_loss = torch.mean(F.mse_loss(self.critic(s), target_td.detach()))

            self.ac_opti.zero_grad()
            ac_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.ac_opti.step()

            self.cr_opti.zero_grad()
            cr_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.cr_opti.step()

    def train(self, episode, show_action=None, save_net=None, show_result=None, ac_file=None, cr_file=None):
        for ep in range(episode):
            buffer_s, buffer_a, buffer_r, buffer_s_ = [], [], [], []
            self.env.start_with_memory()
            # s = self.env.reset()
            self.env.start_with_memory()
            s = self.env.probes
            reward = 0
            self.action = []
            self.rr = []
            # with tqdm(total=(self.env.num_steps//40+1)*40) as pbar:
            #    pbar.set_description('Processing Step_')
            a_ = 0
            for step in range(self.env.num_steps // 40 + 1):
                a = self.get_action(s)
                a = a_ * 0.9 + 0.1 * (a - a_)
                a_ = a
                for _ in range(40):
                    s_, r, done = self.env.evolve(a)
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append((r + 2.9) * 20)
                buffer_s_.append(s_)

                s = s_
                reward += (r + 2.9) * 20
                self.rr.append((r + 2.9) * 10)
                if show_action:
                    self.action.append(a)
                #################self.batch
                if (step + 1) % self.batch == 0 or step == self.env.num_steps // 40:  ##batch_norm&batch_size
                    self.update(buffer_s, buffer_a, buffer_r, buffer_s_)
                    buffer_s, buffer_a, buffer_r, buffer_s_ = [], [], [], []

            if show_action and ep % show_action == 0:
                plt.figure()
                plt.title('Action-Step_ep' + str(ep))
                plt.xlabel('Step')
                plt.ylabel('jet_Q')
                plt.plot(range(len(self.action)), self.action)
                plt.figure()
                plt.title('Action-Step_ep' + str(ep))
                plt.xlabel('Step')
                plt.ylabel('jet_Q')
                plt.plot(range(len(self.rr)), self.rr)
                plt.show()

            if save_net and ep % save_net == 0:
                self.save_model(ac_file, cr_file)

            if show_result and ep % show_result == 0:
                fig = plt.figure()
                f = plt.subplot(1, 2, 1)
                f.set_title('Drag-Step_ep' + str(ep))
                f.set_xlabel('Step')
                f.set_ylabel('Drag')
                plt.plot(range(len(self.env.drag_list)), self.env.drag_list)
                g = plt.subplot(1, 2, 2)
                g.set_title('Lift-Step_ep' + str(ep))
                g.set_xlabel('Step')
                g.set_ylabel('Lift')
                plt.plot(range(len(self.env.lift_list)), self.env.lift_list)
                plt.show()

            self.list_reward.append(reward / (self.env.num_steps // 40 + 1))
            self.list_drag.append(self.env.avg_drag)

            if ep > 0 and ep % 10 == 0:
                self.ac_lr = self.ac_lr * 0.9
                self.cr_lr = self.cr_lr * 0.9





