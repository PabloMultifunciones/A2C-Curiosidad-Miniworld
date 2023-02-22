from worker import worker
from actor_critic import ActorCritic
from icm import ICM
import torch.multiprocessing as mp
import torch
import os

if __name__ == '__main__':   
    n_actions = 3
    num_processes = 1
    save_path_actor_critic = './models/actor_critic'
    save_path_icm = './models/icm'

    env_id = 'MiniWorld-FourRooms-v0'

    input_shape = [4, 42, 42]

    mp.set_start_method('spawn')
    
    global_actor_critic = ActorCritic(input_shape, n_actions)
    global_actor_critic.share_memory()
    global_optim = torch.optim.Adam(global_actor_critic.parameters(), lr=1e-4)

    global_icm = ICM(input_shape, n_actions)
    global_icm.share_memory()
    global_icm_optim = torch.optim.Adam(global_icm.parameters(), lr=1e-4)

    if os.path.isfile(save_path_actor_critic):
        global_actor_critic.load_state_dict(torch.load(save_path_actor_critic))
        print('Cargando ActorCritic')

    if os.path.isfile(save_path_icm):
        global_icm.load_state_dict(torch.load(save_path_icm))
        print('Cargando ICM')

    processes = []
    for pid in range(num_processes):

        p = mp.Process(target=worker, args=(input_shape, n_actions, global_actor_critic, global_optim, global_icm, global_icm_optim, env_id))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    torch.save(global_actor_critic.state_dict(), "./models/actor_critic")
    torch.save(global_icm.state_dict(), "./models/icm")
    

    
