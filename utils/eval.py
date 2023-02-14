import numpy as np

def evaluate(env, agent, ep=0, num_episode=10, video=None, scaler=None):
	"""Evaluate a trained agent and optionally save a video."""
	episode_rewards = []

	for i in range(num_episode):
		ep_reward = 0
		time_step = env.reset()
		if video: video.init(env, enabled=(i==0))
		while not time_step.last():
			action = agent.select_action(time_step.observation, eval_mode=True, t0=time_step.first())

			time_step = env.step(action.cpu().numpy())

			ep_reward += time_step.reward
			if video: video.record(env)
		episode_rewards.append(ep_reward)
		if video: 
			video.save(ep)
	return np.nanmean(episode_rewards)
