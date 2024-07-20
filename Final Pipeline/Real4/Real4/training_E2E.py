from main import TrainEvaluateAgent
import numpy as np
import mpi4py as MPI

class E2E_Training(TrainEvaluateAgent):
    def __init__(self, env_name_, render, real, display, diffusion=False, local=False):
        super().__init__(env_name_, render=render, real=real, display=display, diffusion=diffusion, local=local)

    def reset_environment_E2E(self):
        # while True:
        env_dict = self.env.reset()[0]
        # if np.mean(abs(env_dict["achieved_goal"] - env_dict["desired_goal"])) > 0.02:
        #     return env_dict
        return env_dict

    def eval_agent_E2E(self, randomise):
        # Evaluation logic here
        total_success_rate = []
        running_r = []
        for ep in range(10):
            print("episode: ", ep)
            episode_dict, success, episode_reward, _, _ = self.run_episode_E2E(train_mode=False,
                                                                                randomisation=randomise)
            total_success_rate.append(success[-1])
            if ep == 0:
                running_r.append(episode_reward)
            else:
                running_r.append(running_r[-1] * 0.99 + 0.01 * episode_reward)

        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate)
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size(), running_r, episode_reward

    def train(self, randomise, E2E=False):
        # self.agent.load_weights()
        # Training logic here
        t_success_rate = []
        running_reward_list = []
        total_ac_loss = []
        total_cr_loss = []
        for epoch in range(MAX_EPOCHS):
            start_time = time.time()
            epoch_actor_loss = 0
            epoch_critic_loss = 0
            for cycle in range(0, MAX_CYCLES):
                print("| Epoch: ", epoch, "| Cycle: ", cycle)
                mb = []
                cycle_actor_loss = 0
                cycle_critic_loss = 0
                for episode in range(MAX_EPISODES):
                    # print("Episode: ", episode)
                    if E2E:
                        episode_dict, _, _, _, _ = self.run_episode_E2E(train_mode=True, randomisation=randomise)
                    else:
                        episode_dict, _, _, _, _ = self.run_episode(train_mode=True, randomisation=randomise, E2E=E2E)
                    mb.append(dc(episode_dict))
                # print("store")
                self.agent.store(mb)
                # print("update")
                for n_update in range(num_updates):
                    actor_loss, critic_loss = self.agent.train()
                    cycle_actor_loss += actor_loss
                    cycle_critic_loss += critic_loss

                epoch_actor_loss += cycle_actor_loss / num_updates
                epoch_critic_loss += cycle_critic_loss / num_updates
                self.agent.update_networks()

                if (cycle + 1) % 10 == 0 and cycle != 0:
                    if E2E:
                        success_rate, running_reward, episode_reward = self.eval_agent_E2E(randomise=randomise)
                    else:
                        success_rate, running_reward, episode_reward = self.eval_agent(randomise=randomise)
                    if MPI.COMM_WORLD.Get_rank() == 0:
                        ram = psutil.virtual_memory()
                        t_success_rate.append(success_rate)
                        running_reward_list.append(running_reward)
                        print(f"Epoch:{epoch}| "
                              f"Running_reward:{running_reward[-1]:.3f}| "
                              f"EP_reward:{episode_reward:.3f}| "
                              f"Memory_length:{len(self.agent.memory)}| "
                              f"Duration:{time.time() - start_time:.3f}| "
                              f"Actor_Loss:{actor_loss:.3f}| "
                              f"Critic_Loss:{critic_loss:.3f}| "
                              f"Success rate:{success_rate:.3f}| "
                              f"{to_gb(ram.used):.1f}/{to_gb(ram.total):.1f} GB RAM"
                              )
                        self.agent.save_weights()
                        self.save_data(t_success_rate, running_reward_list)

                    if MPI.COMM_WORLD.Get_rank() == 0:
                        with SummaryWriter("logs") as writer:
                            for i, success_rate in enumerate(t_success_rate):
                                writer.add_scalar("Success_rate", success_rate, i)
                        plt.style.use('ggplot')
                        plt.figure()
                        plt.plot(np.arange(0, epoch * 5 + (cycle + 1) / 10), t_success_rate)
                        plt.title("Success rate")
                        plt.savefig("success_rate.png")
                        # plt.show()

            ram = psutil.virtual_memory()
            # success_rate, running_reward, episode_reward = eval_agent(env, agent)
            total_ac_loss.append(epoch_actor_loss)
            total_cr_loss.append(epoch_critic_loss)
            # if MPI.COMM_WORLD.Get_rank() == 0:
            #     t_success_rate.append(success_rate)
            #     print(f"Epoch:{epoch}| "
            #           f"Running_reward:{running_reward[-1]:.3f}| "
            #           f"EP_reward:{episode_reward:.3f}| "
            #           f"Memory_length:{len(agent.memory)}| "
            #           f"Duration:{time.time() - start_time:.3f}| "
            #           f"Actor_Loss:{actor_loss:.3f}| "
            #           f"Critic_Loss:{critic_loss:.3f}| "
            #           f"Success rate:{success_rate:.3f}| "
            #           f"{to_gb(ram.used):.1f}/{to_gb(ram.total):.1f} GB RAM")
            #     agent.save_weights()

        if MPI.COMM_WORLD.Get_rank() == 0:

            with SummaryWriter("logs") as writer:
                for i, success_rate in enumerate(t_success_rate):
                    writer.add_scalar("Success_rate", success_rate, i)

            plt.style.use('ggplot')
            plt.figure()
            plt.plot(np.arange(0, MAX_EPOCHS), t_success_rate)
            plt.title("Success rate")
            plt.savefig("success_rate.png")
            plt.show()