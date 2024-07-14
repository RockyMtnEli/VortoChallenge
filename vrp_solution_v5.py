import numpy as np
import pandas as pd
from random import shuffle
import sys

MAX_MINS = 12 * 60


def shuffle_solution(sol, dist, r_df, limit):
    old_sol = sol.copy()
    old_dist = dist
    best_sol = sol.copy()
    min_dist = limit
    better = False
    count = 0
    while count < len(sol):
        count += 1
        shuffle(sol)
        dist = compute_total_distance(sol, r_df.values)

        if dist < min_dist:
            better = True
            min_dist = dist
            best_sol = sol.copy()

    if better:
        return best_sol, min_dist
    else:
        return old_sol, old_dist


def check_solution(s_list):
    flat_list = []
    for ss in s_list:
        flat_list.extend(ss)

    tmp = list(set(flat_list))
    if len(tmp) < len(flat_list):
        return False
    else:
        return True


def reset_adv_matrix(s_list, l_list, a_mat):
    new_a_mat = a_mat.copy()
    for ss in s_list:
        for ii in ss:
            new_a_mat[:, ii] = 0.

    for ii in l_list:
        new_a_mat[:, ii] = 0.

    return new_a_mat


def compute_hub_distance(route_arr):
    squares = route_arr * route_arr
    dist_squared_start = squares[:, 0] + squares[:, 1]
    dist_start = np.sqrt(dist_squared_start)
    dist_squared_end = squares[:, 2] + squares[:, 3]
    dist_end = np.sqrt(dist_squared_end)

    return dist_start, dist_end


def compute_distance(route_arr):
    x_diffs = route_arr[:, 2] - route_arr[:, 0]
    y_diffs = route_arr[:, 3] - route_arr[:, 1]
    dist_squared = x_diffs * x_diffs + y_diffs * y_diffs
    dist = np.sqrt(dist_squared)
    
    return dist


def compute_transit_distance(route_arr):
    if len(route_arr) > 1:
        start = route_arr[:-1, 2:]
        end = route_arr[1:, :2]
        transit_arr = np.concatenate((start, end), axis=1)
        dist = compute_distance(transit_arr)
    else:
        dist = 0.

    return dist


def compute_total_distance(path, route_arr):
    path_arr = route_arr[path, :4]
    hub_dist_start, hub_dist_end = compute_hub_distance(path_arr)
    dist = hub_dist_start[0] + hub_dist_end[-1]
    load_dist = compute_distance(path_arr)
    dist += np.sum(load_dist)

    if len(path) > 1:
        transit_dist = compute_transit_distance(path_arr)
        dist += np.sum(transit_dist)

    return dist


def compute_advantage(combined_path, route_arr):
    legs_arr = route_arr[combined_path, :]
    hub_dist_start, hub_dist_end = compute_hub_distance(legs_arr)
    transit_dist = compute_transit_distance(legs_arr)
    adv_dist = 500. + hub_dist_end[0] + hub_dist_start[1] - transit_dist[0]

    return adv_dist


def compute_solution(route_df):
    # start_idx = 0
    tmp_df = route_df.sort_values(by=['pickup_x', 'pickup_y'])
    start_idx = int(list(tmp_df.index)[0])
    del tmp_df

    # initialize the advantage matrix
    adv_matrix = np.zeros((len(route_df), len(route_df)))
    for ii in range(adv_matrix.shape[0]):
        for jj in range(adv_matrix.shape[1]):
            if ii != jj:
                adv_matrix[ii, jj] = compute_advantage([ii, jj], route_df.values)

    orig_adv_matrix = adv_matrix.copy()
    sol_list = []
    dist_list = []
    local_sol = [start_idx]
    total_dist = compute_total_distance(local_sol, route_df.values)
    local_dist_list = [total_dist]
    adv_matrix[:, start_idx] = 0.

    # main path-building loop
    for idx in range(1, len(route_df)):
        local_sol.append(np.argmax(adv_matrix[local_sol[-1], :]))
        total_dist = compute_total_distance(local_sol, route_df.values)
        local_dist_list.append(total_dist)
        if total_dist > MAX_MINS:
            local_sol, total_dist = shuffle_solution(local_sol, total_dist, route_df, MAX_MINS)

            if total_dist <= MAX_MINS:
                local_dist_list.pop(-1)
                local_dist_list.append(total_dist)
                adv_matrix = reset_adv_matrix(sol_list, local_sol, orig_adv_matrix)
            else:
                popped_idx = local_sol.pop(-1)
                sol_list.append(local_sol)
                adv_matrix = reset_adv_matrix(sol_list, local_sol, orig_adv_matrix)
                local_dist_list.pop(-1)
                total_dist = local_dist_list[-1]
                dist_list.append(total_dist)
                local_sol = [popped_idx]
                total_dist = compute_total_distance(local_sol, route_df.values)
                adv_matrix[:, popped_idx] = 0.
        else:
            local_sol, new_total_dist = shuffle_solution(local_sol, total_dist, route_df, total_dist)
            adv_matrix = reset_adv_matrix(sol_list, local_sol, orig_adv_matrix)
            if new_total_dist < total_dist:
                local_dist_list.pop(-1)
                local_dist_list.append(total_dist)

    # add a leftover path if there is one
    if total_dist <= MAX_MINS:
        sol_list.append(local_sol)
        dist_list.append(local_dist_list[-1])

    # try some different path orders
    tmp_sol_list = sol_list.copy()
    while len(dist_list):
        best_option = []
        best_sol = []
        best_dist = MAX_MINS

        for ii in range(1, len(tmp_sol_list)):
            tmp_sol = tmp_sol_list[0] + tmp_sol_list[ii]
            total_dist = compute_total_distance(tmp_sol, route_df.values)
            tmp_sol, total_dist = shuffle_solution(tmp_sol, total_dist, route_df, best_dist)

            if total_dist < best_dist:
                best_dist = total_dist
                best_option = [ii, 0]
                best_sol = tmp_sol

        if len(best_option):
            old_dists = []
            for xx in best_option:
                tmp_sol_list.pop(xx)
                sol_list.pop(xx)
                old_dists.append(dist_list.pop(xx))
            tmp_sol_list.append(best_sol)
            sol_list.append(best_sol)
            dist_list.append(total_dist)
        else:
            tmp_sol_list.pop()
            dist_list.pop()

    # new_sol_list_len = len(sol_list)
    # if new_sol_list_len < orig_sol_list_len:
    #     print(f'\n*** sol list was improved from {orig_sol_list_len} to {new_sol_list_len}\n')
    # else:
    #     print(f'\n*** sol list was not improved from {orig_sol_list_len}\n')

    # good_list = check_solution(sol_list)
    # print(f'solution is good: {good_list}')

    # convert from 0-based index to 1-based
    sol_list_final = []
    for sol in sol_list:
        local_sol = []
        for ii in sol:
            local_sol.append(ii + 1)
        sol_list_final.append(local_sol)

    return sol_list_final


def main():
    file = sys.argv[1]
    # file = './TrainingProblems/problem19.txt'

    # read in the file to a dataframe
    df_in = pd.read_csv(file, sep=' ')

    # convert tuples to columns
    pickup_df = df_in['pickup'].str[1:-1].str.split(',', expand=True).astype(float)
    dropoff_df = df_in['dropoff'].str[1:-1].str.split(',', expand=True).astype(float)

    # combine and rename the columns
    df = pd.concat([pickup_df, dropoff_df], axis=1, ignore_index=True)
    df.rename(columns={0: 'pickup_x', 1: 'pickup_y', 2: 'dropoff_x', 3: 'dropoff_y'}, inplace=True)

    solutions = compute_solution(df)

    for route in solutions:
        print(route)


if __name__ == '__main__':
    main()
