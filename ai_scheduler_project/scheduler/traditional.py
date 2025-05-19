def fcfs(df):
    df = df.sort_values(by="arrival_time").reset_index(drop=True)
    current_time = 0
    schedule = []
    waiting_times = []
    total_burst = 0

    for idx, row in df.iterrows():
        start_time = max(current_time, row['arrival_time'])
        end_time = start_time + row['burst_time']
        waiting_time = start_time - row['arrival_time']
        waiting_times.append(waiting_time)
        total_burst += row['burst_time']

        schedule.append({
            "process_id": f"P{idx+1}",
            "start_time": start_time,
            "end_time": end_time
        })

        current_time = end_time

    avg_wait = sum(waiting_times) / len(waiting_times)
    cpu_util = (total_burst / current_time) * 100  # Handle idle time
    return schedule, {
        "average_waiting_time": round(avg_wait, 2),
        "cpu_utilization": round(cpu_util, 2)
    }


def sjf(df):
    df = df.copy()
    processes = df.to_dict("records")
    completed = []
    schedule = []
    time = 0
    total_burst = 0
    waiting_times = []

    while processes or completed:
        ready_queue = [p for p in processes if p['arrival_time'] <= time]

        if not ready_queue:
            time += 1
            continue

        shortest = min(ready_queue, key=lambda x: x['burst_time'])
        processes.remove(shortest)

        start_time = max(time, shortest['arrival_time'])
        end_time = start_time + shortest['burst_time']
        waiting_time = start_time - shortest['arrival_time']
        waiting_times.append(waiting_time)
        total_burst += shortest['burst_time']

        schedule.append({
            "process_id": f"P{shortest['pid']}" if 'pid' in shortest else f"P{len(completed)+1}",
            "start_time": start_time,
            "end_time": end_time
        })

        shortest['start_time'] = start_time
        shortest['completion_time'] = end_time
        shortest['waiting_time'] = waiting_time

        completed.append(shortest)
        time = end_time

    avg_wait = sum(waiting_times) / len(waiting_times)
    cpu_util = (total_burst / time) * 100

    return schedule, {
        "average_waiting_time": round(avg_wait, 2),
        "cpu_utilization": round(cpu_util, 2)
    }
