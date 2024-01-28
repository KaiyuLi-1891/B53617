from tensorforce.agents import PPOAgent
from tensorforce.execution import Runner

environment = env()

network_spec = [
    dict(type='dense', size=512),
    dict(type='dense', size=512),
]

agent = PPOAgent(
    states=environment.states,
    actions=environment.actions,
    network=network_spec,
    # Agent
    state_preprocessing=None,
    # actions_exploration=None,
    exploration = 0;
    reward_processing = None,
    # reward_estimation[reward_preprocessing]=None, ### Deprecated Agent argument reward_preprocessing, use reward_estimation[reward_preprocessing] instead
    # MemoryModel
    # update_mode=dict(
    #     unit='episodes',
    #     # 10 episodes per update
    #     batch_size=20,
    #     # Every 10 episodes
    #     frequency=20
    # ),
###I reckon that the update_mode function is separated into update_frequency and batch_size in the latest version
    update_frequency = 20,
    memory=dict(
        type='latest',
        include_next_states=False,
        capacity=10000
    ),
    # DistributionModel
    # distributions=None,    #Not sure whether these 2 are the same
    use_beta_distribution = False,
    entropy_regularization=0.01,
    # PGModel
    baseline_mode='states',
    baseline=dict(
        type='mlp',
        sizes=[32, 32]
    ),
    baseline_optimizer=dict(
        type='multi_step',
        optimizer=dict(
            type='adam',
            learning_rate=1e-3
        ),
        num_steps=5
    ),
    gae_lambda=0.97,
    # PGLRModel
    likelihood_ratio_clipping=0.2,
    # PPOAgent
    step_optimizer=dict(
        type='adam',
        learning_rate=1e-3
    ),
    subsampling_fraction=0.2,
    multi_step=25,  ###Deprecated PPO argument optimization_steps, use multi_step instead
    execution=dict(
        type='single',
        session_config=None,
        distributed_spec=None
    ) ,
    saver=dict(
        directory='./saver/', 
        basename='PPO_model.ckpt', 
        load=False, 
        seconds=600),
    max_episode_timesteps = 40000,  ### Required arguments missing
    batch_size = 20,  ### Required arguments missing
)

if(os.path.exists('saved_models/checkpoint')):
    restore_path = './saved_models'
else:
    restore_path = None

if restore_path is not None:
    printi("restore the model")
    agent.restore_model(restore_path)
    
def episode_finished(r):
    name_save = "./saved_models/ppo_model"
    r.agent.save_model(name_save, append_timestep=False)

    # show for plotting
    # r.environment.show_control()
    # r.environment.show_drag()

    # print(sess.run(tf.global_variables()))

    return True
runner = Runner(agent=agent, environment=environment)
runner.run(episodes=2000, max_episode_timesteps=nb_actuations, episode_finished=episode_finished)
runner.close()
