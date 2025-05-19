from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from scheduler.rl_scheduler import run_rl_scheduler  # Your RL scheduler module
from models.ml_burst import predict_burst_time

app = Flask(__name__)

# Load the RandomForest model and other necessary resources
rf_model = joblib.load(r"C:\Users\ayush\Desktop\OS_PBL_Project\AI-Based-CPU-Scheduler-main\approach3\burst_time_prediction\burst_time_predictor.joblib")  # RandomForest model
scaler = joblib.load(r"C:\Users\ayush\Desktop\OS_PBL_Project\AI-Based-CPU-Scheduler-main\approach3\burst_time_prediction\feature_scaler.joblib")  # Feature scaler

# Define the feature names used in the model
FEATURE_NAMES = [
    'io_write_bytes',
    'num_ctx_switches_voluntary',
    'cpu_percent',
    'io_read_bytes',
    'io_read_count',
    'io_write_count'
]

@app.route('/')
def index():
    return render_template('index.html')  # Your HTML form

import logging

# Configure logging to print debug information
logging.basicConfig(level=logging.DEBUG)

@app.route('/schedule', methods=['POST'])
def schedule():
    try:
        # Step 1: Receive data
        data = request.get_json()

        # Log received data to ensure it is correct
        logging.debug(f"Received data: {data}")

        processes = data['processes']
        algorithm = data['algorithm']

        # Step 2: Predict burst times for the incoming processes
        for process in processes:
            process_features = [
                process['io_write_bytes'],
                process['num_ctx_switches_voluntary'],
                process['cpu_percent'],
                process['io_read_bytes'],
                process['io_read_count'],
                process['io_write_count']
            ]
            predicted_burst_time = predict_burst_time(process_features)
            process['predicted_burst_time'] = predicted_burst_time

        # Step 3: Select scheduling algorithm (e.g., FCFS, SJF, RL)
        logging.debug(f"Scheduling algorithm selected: {algorithm}")

        if algorithm == 'fcfs':
            result = fcfs_scheduler(processes)
        elif algorithm == 'sjf':
            result = sjf_scheduler(processes)
        elif algorithm == 'rl':
            # Format process list to include 'features' key for RL scheduler
            formatted_processes = []
            for proc in processes:
                features = [
                    proc['io_write_bytes'],
                    proc['num_ctx_switches_voluntary'],
                    proc['cpu_percent'],
                    proc['io_read_bytes'],
                    proc['io_read_count'],
                    proc['io_write_count']
                ]
                formatted_processes.append({
                    'arrival_time': proc['arrival_time'],
                    'features': features,
                    'pid': proc['pid']
                })

            result = run_rl_scheduler(formatted_processes)

        else:
            raise ValueError(f"Unknown algorithm selected: {algorithm}")

        logging.debug(f"Scheduling result: {result}")

        # Step 4: Return scheduling results
        return jsonify(result)
    
    except Exception as e:
        logging.error(f"Error during scheduling: {str(e)}")
        return jsonify({'error': str(e)}), 500




def fcfs_scheduler(processes):
    # Simple FCFS scheduling based on arrival times
    processes.sort(key=lambda p: p['arrival_time'])
    
    current_time = 0
    completed_processes = []
    
    for proc in processes:
        completion_time = current_time + proc['predicted_burst_time']
        turnaround_time = completion_time - proc['arrival_time']
        waiting_time = turnaround_time - proc['predicted_burst_time']
        
        completed_processes.append({
            'pid': proc['pid'],
            'arrival_time': proc['arrival_time'],
            'burst_time': proc['predicted_burst_time'],
            'completion_time': completion_time,
            'turnaround_time': turnaround_time,
            'waiting_time': waiting_time
        })
        
        current_time = completion_time
    
    avg_waiting_time = np.mean([p['waiting_time'] for p in completed_processes])
    return {
        'schedule': completed_processes,
        'avg_waiting_time': avg_waiting_time
    }

def sjf_scheduler(processes):
    # Shortest Job First Scheduling
    processes.sort(key=lambda p: p['predicted_burst_time'])
    
    current_time = 0
    completed_processes = []
    
    for proc in processes:
        completion_time = current_time + proc['predicted_burst_time']
        turnaround_time = completion_time - proc['arrival_time']
        waiting_time = turnaround_time - proc['predicted_burst_time']
        
        completed_processes.append({
            'pid': proc['pid'],
            'arrival_time': proc['arrival_time'],
            'burst_time': proc['predicted_burst_time'],
            'completion_time': completion_time,
            'turnaround_time': turnaround_time,
            'waiting_time': waiting_time
        })
        
        current_time = completion_time
    
    avg_waiting_time = np.mean([p['waiting_time'] for p in completed_processes])
    return {
        'schedule': completed_processes,
        'avg_waiting_time': avg_waiting_time
    }



if __name__ == '__main__':
    app.run(debug=True)
