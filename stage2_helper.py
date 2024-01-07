import numpy as np

def polar2cmplx(r, theta):
    """
    Creates a complex number in the format a+bj when radius and angle in polar form are given.
    :param r: absolute value/modulus/magnitude of the complex number
    :param theta: argument/angle of the complex number

    Returns the corresponding complex number (in the format a+bj). 
    """
    return r * np.exp((0+1j) * theta)

def train(model_, env_, audio_num, max_num_steps, show_effect=True, reward_history=None, action_history=None):
    """
    Trains a model in a given environment over a specified number of time steps.
    :param model_: the DRL model 
    :param env_: the environment the model is going to trained in
    :param audio_num: the number/index of the audio track to be used as the training data
    :param max_num_steps: the maximum number of time steps for training

    Returns None
    """

    if reward_history == None: reward_history = [][:]
    if action_history == None: action_history = []
    step_count = 0

    # reset the environment
    state, _ = env_.reset(options={'reset_all': True, 'audio_num':audio_num, 'show_effect': show_effect}) # FOR SINGLE EPISODE
    done = False

    while not done:
        # feed the state to the agent (model) and get an action
        action = model_.choose_action(state).numpy() # this includes the exploration noise

        # take the action in the environment
        next_state, reward, terminated, truncated, _ = env_.step(action)
        done = terminated | truncated
        step_count += 1

        # store the transition in the replay buffer of the DDPG agent
        model_.remember(state, action, reward, next_state, done)

        # train the model
        model_.learn()

        # set the `next_state` as `state`
        state = next_state

        # keep track of `reward` and `action`
        reward_history.append(reward)
        action_history.append(action)
        
        if step_count >= max_num_steps:
            done = True

    return reward_history, action_history