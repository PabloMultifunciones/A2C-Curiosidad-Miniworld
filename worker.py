import numpy as np
import torch as T
from memory import Memory
from utils import plot_learning_curve 
from wrappers import make_env
from actor_critic import ActorCritic
from icm import ICM

def worker(input_shape, n_actions, global_agent, global_agent_optimizer, global_icm, global_icm_optimizer, env_id):
    T_MAX = 20
    
    frame_buffer = [input_shape[1], input_shape[2], 1]
    env = make_env(env_id, shape=frame_buffer)

    local_agent = ActorCritic(input_shape, n_actions)
    local_icm = ICM(input_shape, n_actions)

    memory = Memory()

    episode, max_steps, t_steps, scores = 0, 20000, 0, []
    
    while episode < max_steps:
        state = env.reset()

        score, done, ep_steps = 0, False, 0
        hx = T.zeros(1, 256)

        while not done:
            env.render()
            action, value, log_prob, hx = local_agent(state, hx)

            next_state, reward, done, _ = env.step(action)

            memory.remember(state, action, next_state, reward, value, log_prob)

            score += reward
            state = next_state
            ep_steps += 1
            t_steps += 1

            if ep_steps % T_MAX == 0 or done:
                states, actions, new_states, rewards, values, log_probs = memory.sample_memory()

                intrinsic_reward, L_I, L_F = local_icm.calc_loss(states, new_states, actions)

                loss = local_agent.calc_cost(state, hx, done, rewards, values, log_probs, intrinsic_reward)

                global_agent_optimizer.zero_grad()
                global_icm_optimizer.zero_grad()

                hx = hx.detach()

                (L_I + L_F).backward()
                loss.backward()

                T.nn.utils.clip_grad_norm_(global_agent.parameters(), 40)
                
                for local_param, global_param in zip(local_agent.parameters(), global_agent.parameters()):
                    global_param._grad = local_param.grad

                for local_param, global_param in zip(local_icm.parameters(), global_icm.parameters()):
                    global_param._grad = local_param.grad

                global_agent_optimizer.step()
                global_icm_optimizer.step()

                local_agent.load_state_dict(global_agent.state_dict())
                local_icm.load_state_dict(global_icm.state_dict())

                memory.clear_memory()

        episode += 1
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        avg_score_5000 = np.mean(scores[max(0, episode-5000): episode+1])
        print('ICM episode {} steps {:.2f}M score {:.2f} avg score (100) (5000) {:.2f} {:.2f}'.format(episode, t_steps/1e6, score, avg_score, avg_score_5000))
    x = [z for z in range(episode)]
    plot_learning_curve(x, scores, 'ICM_hallway_final.png')
