from tensorforce.environments import Environment

class env(Environment, Env2DAirfoil):
    def __init__(self):
        self.reset_flag = 0
        # super().__init__()
        Environment.__init__(self)
        Env2DAirfoil.__init__(self)

        self.states = dict(type='float', shape=(self.probes_num*2,))##velocity

        self.actions = dict(type='float',
                    shape=(1,),##jet_num
                    min_value=-0.02,
                    max_value=0.02)
        

    def states(self):
        return dict(type='float', shape=(self.probes_num*2,))##velocity

    def actions(self):
        return dict(type='float',
                    shape=(1,),##jet_num
                    min_value=-0.02,
                    max_value=0.02)

    # Optional additional steps to close environment
    def close(self):
        super().close()

    def reset(self):
        memorize_state()###?
        state = self.env.probes ##memstate, randomstate
        return state##memstate

    def execute(self, actions):
        #actions_list = actions.numpy().transpose().tolist()[0]
        #Q1, Q2=actions[0], actions[1]
        #Q3 = -Q1 - Q2
        next_state, reward, terminal = evolve(actions)#(Q1, Q2, Q3)
        return next_state, terminal, reward
