import sys
import numpy as np

import gymnasium as gym
from gymnasium import spaces

sys.path.append('../')
from stage1_helper import apply_filter, SNR, create_target_and_jammed_signals


class UniStateReceiverEnv(gym.Env):
    """
    A custom environment developed in the accordance with gym environment API that immitates a single state environment. 
    i.e., the objective of the agent is going to be find the best filter that maximizes the reward for a fixed signal partition. 
    :param N: FIR filter length, must be an odd integer 
    :param S: buffer size at the receiver
    """

    # define constants 
    MIN_BUFFER_SIZE = 20
    EPISODE_LENGTH  = 100

    def __init__(self, N:int, S:int):
        super(UniStateReceiverEnv, self).__init__()

        # ----- verifying input arguments and setting them as class atributes ----
        # filter length 
        if N%2 != 1:
            raise Exception(f"FIR filter length 'N' must be an odd integer: given {N}")
        self.N = N

        # signal partition size
        if S < self.MIN_BUFFER_SIZE:
            raise Exception(f"the buffer size 'S' must be larger than the MIN_BUFFER_SIZE, {self.MIN_BUFFER_SIZE}: given {S}")
        self.S = S
        # buffer size 
        self.buffer_size = S + N - 1

        # ----------------------------- Action Space -----------------------------
        # action - choosing the filter coefficients from index 0 to (N-1)/2; 
        # NOT TUNING/ADJUSTING/CHANGING the coefficeints of an existing filter that the agent is not aware of 
        action_shape = (int((N+1)/2), )
        self.action_space = spaces.Box(low=-1, high=1, shape=action_shape, dtype=np.float32) # float16 -> float32

        # ----------------------------- State Space ------------------------------
        state_shape = (self.buffer_size, )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=state_shape, dtype=np.int32)

        # ------------------------- other class atributes ------------------------
        target_signal, jammed_signal = create_target_and_jammed_signals('vignesh', truncation_freq=5_000, interference_center_freq=12_000)
        self.state = jammed_signal[:self.buffer_size] # fixed state
        self.counter = 0 # a counter to keep track of the number of elapsed time steps of the environment

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=None)

        # set the agent back to the initial state 
        # in this case, we have only one state in the entire environment
        initial_state = self.state

        if options == 'reset_counter':
            self.counter = 0

        info = {}

        # return the initial state
        # make sure that the state returned by env.reset() is same as the end state of the previous episode to emulate a non-episodic agent-env interaction. 
        return initial_state, info

    def step(self, action):

        # increment the counter
        self.counter += 1

        # create the filter 
        filter = np.concatenate((action[-1:0:-1], action))

        # received signal partition in the buffer
        partition = self.state

        # apply the filter 
        filtered = apply_filter(filter, partition)[(self.N-1)//2 : (self.N-1)//2 + self.S]
        target   = self.target_signal[(self.N-1)//2 : (self.N-1)//2 + self.S]

        # calculate the reward (SNR)
        reward = SNR(target, filtered)
        if np.isnan(reward):
            raise Exception("reward value is not a number...")
        
        # truncating the episode
        truncated = False
        if self.counter % self.EPISODE_LENGTH == 0:
            truncated = True
        
        next_state = self.state 
        terminated = False
        info = {}
        
        return next_state, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass
