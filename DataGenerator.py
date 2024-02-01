import csv
import multiprocessing
import random
import sys
import threading

from tqdm import tqdm
import pandas as pd
import numpy as np
import Ramsey_ExperimentV2

# total_experiments = 1
total_time = 0.5 * np.pi


# total_time_stamps = 2
# number_of_qubits = 2
# shots = 500
# # time_stamps = [0.0922, 0.35693]
# time_stamps = np.linspace(0, total_time, total_time_stamps)
# file_name = 'experiments.csv'


def run_experiment(qubits, total_experiments, total_time_stamps, shots, mean_decay, filename='experiments.csv'):
    experiments = []

    time_stamps = np.linspace(0, total_time, total_time_stamps)

    def create_csv_from_experiments(experiments, decay, W, J, filename):
        # Open the file in write mode
        filename = "Data/" + filename + ".csv"
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            # Create a CSV writer
            csv_writer = csv.writer(file)

            # Prepare the header
            header = []
            max_length = max(len(experiment) for experiment in experiments)
            for i in range(max_length):
                for j in range(len(experiments[0][i])):
                    header.append(f'{i}_{j}')
            for i in range(len(decay[0])):
                header.append(f'decay_{i}')
            for i in range(len(W[0])):
                header.append(f'W_{i}')
            for i in range(len(J[0])):
                header.append(f'J_{i}')

            # Write the header
            csv_writer.writerow(header)

            # Write each experiment's data
            for j, experiment in enumerate(experiments):
                row = []
                for part in experiment:
                    row.extend(part)
                for k in decay[j]:
                    row.append(k)
                for k in W[j]:
                    row.append(k)
                for k in J[j]:
                    row.append(k)
                csv_writer.writerow(row)

    W_parameters = []
    J_parameters = []
    decay_parameters = []
    with tqdm(total=total_experiments, file=sys.stdout, dynamic_ncols=True, desc=f'Experiments for {filename}') as pbar:
        for i in range(total_experiments):
            experiment_parts = []
            L = [random.gauss(mean_decay, 2) for _ in range(qubits)]
            W = [random.gauss(5, 2) for _ in range(qubits)]
            J = [random.gauss(0, 0) for _ in range(qubits - 1)]
            W_parameters.append(W)
            J_parameters.append(J)
            decay_parameters.append(L)
            for t in time_stamps:
                exp = Ramsey_ExperimentV2.RamseyExperiment(qubits, t, shots, J, W, L)
                exp.create_full_circuit()
                exp.add_decay_raw()
                # exp.add_noise_raw()
                values = exp.get_z_nearest_neighbors()
                experiment_parts.append(values)

            experiments.append(experiment_parts)
            pbar.update(1)
    create_csv_from_experiments(experiments, decay_parameters, W_parameters, J_parameters, filename)


# run_experiment(number_of_qubits, time_stamps, shots, total_experiments, filename=file_name)


def read_excel_to_variables(file_path):
    # Read the Excel file into a DataFrame
    df = pd.read_excel(file_path)

    # Create a dictionary to hold the variables
    variables = {}

    # Iterate through each column and assign it to a variable
    for column in df.columns:
        variables[column] = df[column].values

    return variables


# File path for the uploaded Excel file
file_path = 'Data_generator_template.xlsx'
threads = []

# Read and store the data from the Excel file
variables = read_excel_to_variables(file_path)


#
# for i in range(len(variables["Qubits"])):
#     run_experiment(variables["Qubits"][i], variables["Total_Experiments"][i], variables["Time_Stamps"][i],
#                    variables["Shots"][i], variables["Mean_Decay"][i], filename=variables["File_Name"][i])
#

def process_experiment(args):
    run_experiment(*args)


def main():
    # Prepare the arguments for each experiment
    experiment_args = [
        (
            variables["Qubits"][i],
            variables["Total_Experiments"][i],
            variables["Time_Stamps"][i],
            variables["Shots"][i],
            variables["Mean_Decay"][i],
            variables["File_Name"][i]
        )
        for i in range(len(variables["Qubits"]))
    ]

    # Create a list to hold the processes
    processes = []

    # Create and start a process for each set of arguments
    for args in experiment_args:
        process = multiprocessing.Process(target=process_experiment, args=(args,))
        processes.append(process)
        process.start()

    # Wait for all processes to complete
    for process in processes:
        process.join()

    print("Done!")


if __name__ == '__main__':
    main()
