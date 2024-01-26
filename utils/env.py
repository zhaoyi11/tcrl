from collections import deque, defaultdict, OrderedDict
from typing import Any, NamedTuple
import dm_env
import numpy as np
from dm_control import suite
from dm_control.suite.wrappers import action_scale
from dm_env import StepType, specs
from dm_control.suite.wrappers import action_scale, pixels

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    pixels: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)
            
class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key='pixels'):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()
        assert pixels_key in wrapped_obs_spec

        pixels_shape = wrapped_obs_spec[pixels_key].shape
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = specs.BoundedArray(shape=np.concatenate(
            [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
                                            dtype=np.uint8,
                                            minimum=0,
                                            maximum=255,
                                            name='observation')

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)
    
    def _transform_reward(self, time_step):
        assert len(self._rewards) <= self._num_frames
        while len(self._rewards) < self._num_frames:
            self._rewards.append(time_step.reward)

        r = (np.array(list(self._rewards)) * self._reward_weight).sum() # weigheted sum over stacking rewards
        return time_step._replace(reward=r)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation[self._pixels_key]
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
            # self._rewards.append(0)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        time_step = self._transform_observation(time_step)
        return time_step

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)
    
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                            dtype,
                                            wrapped_action_spec.minimum,
                                            wrapped_action_spec.maximum,
                                            'action')

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        if 'pixels' in time_step.observation:
            pixels = time_step.observation.pop('pixels')
            pixels = pixels.transpose(2, 0, 1)
            return ExtendedTimeStep(observation=time_step.observation,
                                pixels=pixels,
                                step_type=time_step.step_type,
                                action=action,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0) 
        else:
            return ExtendedTimeStep(observation=time_step.observation,
                                    pixels=None,
                                    step_type=time_step.step_type,
                                    action=action,
                                    reward=time_step.reward or 0.0,
                                    discount=time_step.discount or 1.0)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)


class ConcatObsWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env
        init_time_step = self._concate_obs(self._env.reset())
        self.obs_shape = init_time_step.observation.shape

    def step(self, action):
        time_step = self._env.step(action)
        return self._concate_obs(time_step)

    def reset(self):
        time_step = self._env.reset()
        return self._concate_obs(time_step)

    def observation_spec(self):
        return dm_env.specs.Array(shape=self.obs_shape, dtype='float64', name='observation')

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _concate_obs(self, time_step):
        obs = time_step.observation
        return time_step._replace(observation=np.concatenate([v.flatten() for v in obs.values()]))


def make_env(env_name, seed, action_repeat, record_video):
    """
    Make environment for TCRL experiments.
    Adapted from https://github.com/facebookresearch/drqv2
    """
    domain, task = str(env_name).replace('-', '_').split('_', 1)
    domain = dict(cup='ball_in_cup', mass='point_mass').get(domain, domain)

    if (domain, task) in suite.ALL_TASKS:
        env = suite.load(domain,
                        task,
                        task_kwargs={'random': seed},
                        visualize_reward=False)
    else:
        raise ValueError
    
    if record_video:
        # zoom in camera for quadruped
        camera_id = dict(quadruped=2).get(domain, 0)
        render_kwargs = dict(height=64, width=64, camera_id=camera_id)
        env = pixels.Wrapper(env,
                                pixels_only=False,
                                render_kwargs=render_kwargs)

    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = ExtendedTimeStepWrapper(env)
    env = ConcatObsWrapper(env)

    return env
    