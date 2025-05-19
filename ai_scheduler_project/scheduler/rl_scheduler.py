import gym
import numpy as np
from gym import spaces
import joblib
from stable_baselines3 import PPO
import pandas as pd
import os

class Process:
    def __init__(self, pid, arrival_time, features):
        self.pid = pid
        self.arrival_time = arrival_time
        self.features = features
        self.waiting_time = 0
        self.execution_time = 0
        self.completed = False
        self.predicted_burst_time = None
        self.completion_time = None  

class ProcessSchedulingEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # Load the ML burst time predictor
        self.burst_predictor = joblib.load(r"C:\Users\ayush\Desktop\OS_PBL_Project\ai_scheduler_project\models\burst_time_predictor.joblib")
        self.scaler = joblib.load(r"C:\Users\ayush\Desktop\OS_PBL_Project\ai_scheduler_project\models\feature_scaler.joblib")

        self.FEATURE_NAMES = [
            'io_write_bytes',
            'num_ctx_switches_voluntary',
            'cpu_percent',
            'io_read_bytes',
            'io_read_count',
            'io_write_count'
        ]

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([100, 100, 20, 100], dtype=np.float32),
            dtype=np.float32
        )

        self.current_time = 0
        self.ready_queue = []
        self.running_process = None
        self.completed_processes = []
        self.total_processes = 0

        self.COMPLETION_REWARD = 5.0
        self.SWITCH_PENALTY = -0.5
        self.WAIT_PENALTY = -0.1
        self.TIME_QUANTUM = 4
        self.LONG_WAIT_THRESHOLD = 50

    def predict_burst_time(self, features):
        features_df = pd.DataFrame([features], columns=self.FEATURE_NAMES)
        scaled_features = pd.DataFrame(
            self.scaler.transform(features_df),
            columns=self.FEATURE_NAMES
        )
        return self.burst_predictor.predict(scaled_features)[0]

    def get_state(self):
        if not self.running_process:
            return np.zeros(4, dtype=np.float32)

        execution_progress = self.running_process.execution_time / max(1, self.running_process.predicted_burst_time)

        return np.array([
            min(100, self.running_process.predicted_burst_time),
            min(100, self.running_process.waiting_time),
            min(20, len(self.ready_queue)),
            min(100, execution_progress * 100)
        ], dtype=np.float32)

    def step(self, action):
        reward = 0
        done = False

        if action == 1 and len(self.ready_queue) > 0:
            if self.running_process:
                if self.running_process.execution_time > self.TIME_QUANTUM:
                    reward += self.SWITCH_PENALTY * 0.5
                else:
                    reward += self.SWITCH_PENALTY
                self.ready_queue.append(self.running_process)
            self.running_process = self.ready_queue.pop(0)

        if self.running_process:
            self.running_process.execution_time += 1

            if self.running_process.execution_time >= self.running_process.predicted_burst_time:
                self.running_process.completion_time = self.current_time + 1  # ✅ Set completion time
                self.completed_processes.append(self.running_process)
                efficiency = max(0.2, 1.0 - (self.running_process.waiting_time / self.LONG_WAIT_THRESHOLD))
                reward += self.COMPLETION_REWARD * efficiency

                self.running_process = self.ready_queue.pop(0) if self.ready_queue else None

        queue_length = len(self.ready_queue)
        for process in self.ready_queue:
            process.waiting_time += 1
            if process.waiting_time > self.LONG_WAIT_THRESHOLD:
                reward += self.WAIT_PENALTY * 2
            else:
                reward += self.WAIT_PENALTY * (queue_length / 10)

        self.current_time += 1

        if not self.running_process and not self.ready_queue:
            done = True
            if len(self.completed_processes) == self.total_processes:
                avg_waiting_time = np.mean([p.waiting_time for p in self.completed_processes])
                completion_bonus = 10.0 * (1.0 - min(1.0, avg_waiting_time / self.LONG_WAIT_THRESHOLD))
                reward += completion_bonus

        return self.get_state(), float(reward), done, {}

    def reset(self):
        self.current_time = 0
        self.completed_processes = []
        self.ready_queue = []
        self.running_process = None

        self.total_processes = np.random.randint(5, 15)
        for i in range(self.total_processes):
            features = [
                np.random.randint(1000, 100000),
                np.random.randint(10, 1000),
                np.random.uniform(0, 100),
                np.random.randint(1000, 100000),
                np.random.randint(100, 1000),
                np.random.randint(100, 1000)
            ]

            process = Process(pid=i, arrival_time=self.current_time, features=features)
            process.predicted_burst_time = self.predict_burst_time(features)
            self.ready_queue.append(process)

        if self.ready_queue:
            self.running_process = self.ready_queue.pop(0)

        return self.get_state()

def train_rl_scheduler(total_timesteps=150000):
    env = ProcessSchedulingEnv()
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0001,
        n_steps=1024,
        batch_size=32,
        n_epochs=8,
        gamma=0.99,
        verbose=1
    )

    try:
        model.learn(total_timesteps=total_timesteps)
        model.save("rl_scheduler_model")
        return model
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return None

def evaluate_scheduler(model, episodes=10):
    env = ProcessSchedulingEnv()

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action, _ = model.predict(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward

        avg_waiting_time = np.mean([p.waiting_time for p in env.completed_processes])
        completion_time = env.current_time

        print(f"\nEpisode {episode + 1} Results:")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Average Waiting Time: {avg_waiting_time:.2f}")
        print(f"Total Completion Time: {completion_time}")
        print(f"Processes Completed: {len(env.completed_processes)}/{env.total_processes}")

        # Show completion time for each process
        print("\nPer-process Completion Times:")
        for p in sorted(env.completed_processes, key=lambda x: x.pid):
            print(f"Process {p.pid}: Completed at time {p.completion_time}, Waiting Time: {p.waiting_time}")





from stable_baselines3 import PPO
import logging

# Configure logging to print debug information
logging.basicConfig(level=logging.DEBUG)

def run_rl_scheduler(process_list):
    try:
        # Initialize the environment
        logging.debug("Initializing RL environment")
        env = ProcessSchedulingEnv()

        # Reset environment state
        env.reset()

        logging.debug(f"Environment state after reset: {env.get_state()}")

        # Clear existing processes in the environment
        env.ready_queue = []
        env.completed_processes = []
        env.running_process = None
        env.total_processes = len(process_list)
        env.cpu_active_time = 0  # NEW: To track CPU time

        logging.debug(f"Number of processes: {env.total_processes}")

        # Load processes into the environment
        for i, process_data in enumerate(process_list):
            arrival_time, features = process_data['arrival_time'], process_data['features']
            process = Process(pid=i, arrival_time=arrival_time, features=features)
            process.predicted_burst_time = env.predict_burst_time(features)
            env.ready_queue.append(process)

            logging.debug(f"Added process {process.pid}: Arrival Time: {arrival_time}, Features: {features}")

        # If there are processes, set the first one to be the running process
        if env.ready_queue:
            env.running_process = env.ready_queue.pop(0)
            logging.debug(f"Running process set: {env.running_process.pid}")

        # Load the pre-trained RL model
        logging.debug("Loading pre-trained RL model")
        model = PPO.load(r"C:\Users\ayush\Desktop\OS_PBL_Project\ai_scheduler_project\models\rl_scheduler_model.zip")
        logging.debug("RL model loaded successfully")

        # Start RL scheduling loop
        state = env.get_state()
        done = False
        total_reward = 0

        logging.debug("Starting scheduling using RL model")
        while not done:
            action, _ = model.predict(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward

            logging.debug(f"State: {state}, Action: {action}, Reward: {reward}, Done: {done}")

        # Finalize process metrics
                # Finalize process metrics
        for process in env.completed_processes:
            process.turnaround_time = process.completion_time - process.arrival_time
            process.waiting_time = max(process.turnaround_time - process.predicted_burst_time, 0)

        total_time = env.current_time if hasattr(env, 'current_time') else max(p.completion_time for p in env.completed_processes)
        cpu_active_time = sum(p.predicted_burst_time for p in env.completed_processes)
        cpu_utilization = round((cpu_active_time / total_time) * 100, 2) if total_time > 0 else 0.0

        avg_waiting_time = round(np.mean([p.waiting_time for p in env.completed_processes]), 2)
        avg_turnaround_time = round(np.mean([p.turnaround_time for p in env.completed_processes]), 2)
        avg_completion_time = round(np.mean([p.completion_time for p in env.completed_processes]), 2)
        throughput = round(len(env.completed_processes) / total_time, 2) if total_time > 0 else 0.0

        # Gather final results
        results = {
            'schedule': [],
            'avg_waiting_time': avg_waiting_time,
            'avg_turnaround_time': avg_turnaround_time,
            'avg_completion_time': avg_completion_time,
            'throughput': throughput,
            'cpu_utilization': f"{cpu_utilization}%",
            'features_used': [process.features for process in env.completed_processes]
        }


        for process in env.completed_processes:
            results['schedule'].append({
                'pid': process.pid,
                'arrival_time': process.arrival_time,
                'burst_time': round(process.predicted_burst_time, 2),
                'completion_time': round(process.completion_time, 2),
                'turnaround_time': round(process.turnaround_time, 2),
                'waiting_time': round(process.waiting_time, 2)
            })

        logging.debug(f"Final results: {results}")
        return results

    except Exception as e:
        logging.error(f"Error during RL scheduling: {str(e)}")
        return {'error': str(e)}






if __name__ == "__main__":
    print("Training RL Scheduler...")
    model = train_rl_scheduler()

    print("\nEvaluating RL Scheduler...")
    evaluate_scheduler(model)
