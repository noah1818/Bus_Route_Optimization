# -----------------------------------------------------------------------------
# Project:     Bus Route Optimization
# File:        main.py
# Company:     Stratton Oakmont
# Date:        September 13, 2024
# Version:     2.2.3
# Description: This script contains the main file to start the model with your settings.
# -----------------------------------------------------------------------------
# License:     Proprietary - Stratton Oakmont. All rights reserved.
# -----------------------------------------------------------------------------
from model import Model
from data_preperation import Data_Preparator
from dataset_creation import DataSetCreator

# path to the data frames
path_data_all = "/Users/noah/Downloads/data-all.csv"
path_route_all = "/Users/noah/Downloads/route-all.csv"
path_graph_all = "/Users/noah/Downloads/graph.csv"
path_route_all_missing_last_day = "/Users/noah/Downloads/route-all-missing-last-day.csv"

if 'main' in __name__:
    # create instance of the data helper class
    data_preparator = Data_Preparator()
    route_all_missing_last_day = data_preparator.get_cleaned_route_all_missing_last_day_df(
        path_route_all_missing_last_day)
    route_all = data_preparator.get_cleaned_route_all_df(path_route_all)
    data_all = data_preparator.get_cleaned_data_all_df(
        path_data_all, route_all_missing_last_day)
    graph_all = data_preparator.get_cleaned_graph_df(path_graph_all)

    # this creates an instance of the model
    # Note: Before using the model we create the Datasets for the model, like rush hours, no rush hours etc...
    dataset_creator = DataSetCreator()
    seasonal_similarities = dataset_creator.seasonal_similarities_df()

    no_rh_night_df = dataset_creator.create_no_rh_night(
        data_all, seasonal_similarities['no_rh_night']['hours'], seasonal_similarities['no_rh_night']['weekdays'])
    no_rh_day_df = dataset_creator.create_no_rh_day(
        data_all, seasonal_similarities['no_rh_day']['hours'], seasonal_similarities['no_rh_day']['weekdays'])
    morning_rh_df = dataset_creator.create_morning_rh(
        data_all, seasonal_similarities['morning_rh']['hours'], seasonal_similarities['morning_rh']['weekdays'])
    evening_rh_df = dataset_creator.create_evening_rh(
        data_all, seasonal_similarities['evening_rh']['hours'], seasonal_similarities['evening_rh']['weekdays'])
    weekend_df = dataset_creator.create_weekend_df(
        data_all, seasonal_similarities['weekend']['hours'], seasonal_similarities['weekend']['weekdays'])

    # here you can initilaize the model by choosing your start and end node, as well as the amount of similar scenarios you want to take into account
    # (see our paper for details regarding the amount of similar scenarios)
    n_similar = 10
    start = 94
    end = 162

    txt_path = "/Users/noah/multidim_optimierung/routes.csv"
    model_args = {
        'data_all': data_all,
        'route_all': route_all_missing_last_day,
        'graph_all': graph_all,
        'no_rh_night_df': no_rh_night_df,
        'no_rh_day_df': no_rh_day_df,
        'morning_rh_df': morning_rh_df,
        'evening_rh_df': evening_rh_df,
        'weekend_df': weekend_df,
        'time_columns': 9,
        'n_similar': n_similar,
        'start': start,
        'end': end
    }
    # create an instance of the model
    model = Model(**model_args)

    # start the algo with your scenario and the start and end node, as well as the scenario method and the heuristic for the A*
    # here you can choose between euclidean_distance, rbf kernel and manhatten distance
    scenario_method = 'euclidean_distance'
    # here you can choose the heuristic used in the A* algorithm.
    heuristic_a_star = 1
    scenario = 0

    path, cost = model.start_algo(scenario, scenario_method, heuristic_a_star)
