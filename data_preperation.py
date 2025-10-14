# -----------------------------------------------------------------------------
# Project:     Bus Route Optimization
# File:        data_preperation.py
# Company:     Stratton Oakmont
# Date:        September 13, 2024
# Version:     2.2.3
# Description: This script is used by the main.py file to make some data preperation.
# -----------------------------------------------------------------------------
import pandas as pd
import csv
import numpy as np


class Data_Preparator(object):
    def __init__(self) -> None:
        pass

    def get_max_len(self, path: str) -> int:
        """
        Returns the maximum length of all rows in the text file.

        Parameters
        ----------
            path : Path to the text file.

        Returns
        -------
            int : The maximum length.
        """

        with open(path, 'r') as file:
            reader = csv.reader(file)
            max_len = max(len(row) for row in reader)
        return max_len

    def route_all_df_helper(self, path: str, column_names: list, max_len: int) -> pd.DataFrame:
        """
        Creates a DataFrame from a CSV file, filling missing values with zeros to ensure consistent row lengths.

        This function reads a CSV file line by line, processes each line to ensure it has the maximum specified number
        of columns (max_len), and fills any missing values with zeros. The last three values of each row are assumed to 
        be day, month, and timestamp, and these are always preserved at the end of the row. The resulting DataFrame has 
        consistent row lengths, with missing values filled as zeros.

        Parameters
        ----------
            path : The path to the CSV file.
            column_names : List of column names for the DataFrame.
            max_len : The maximum length of the columns.

        Returns
        -------
            pd.DataFrame : A DataFrame with consistent row lengths, filled with zeros where necessary.
        """

        rows = []
        with open(path, 'r') as file:
            for line in file:
                try:
                    line_split = line.strip().split(',')
                    N = max_len - len(line_split)
                    row = line_split[:len(line_split)-3] + \
                        [-1] * N + line_split[len(line_split)-3:]
                    rows.append(row)
                except Exception as e:
                    continue
        df = pd.DataFrame(rows, columns=column_names)
        return df

    def convert_first_n_columns_to_float(self, df: pd.DataFrame, N: int) -> pd.DataFrame:
        """
        Converts the values of the first N columns in a DataFrame to float.
        This function takes a DataFrame and converts the values in the first N columns to float type. 
        It modifies the DataFrame in place and returns the modified DataFrame.

        Parameters
        -------
            df : The DataFrame to modify.
            N : The number of the first columns to convert to float.

        Returns
        -------
            pd.DataFrame : The DataFrame with the first N columns converted to float type.
        """

        df.iloc[:, :N] = df.iloc[:, :N].astype(float)
        return df

    def split_times(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Splits a time string in the format 'hh_mm_ss' into separate 'hours', 'minutes', and 'seconds' columns.

        Parameters
        ----------
            df : The DataFrame containing a 'hh_mm_ss' column, where each entry is a string representing time in the format 'hh_mm_ss'.

        Returns
        -------
            pd.DataFrame : The modified DataFrame with separate 'hours', 'minutes', and 'seconds' columns and without the 'hh_mm_ss' column.
        """

        time_split = df['hh_mm_ss'].str.split('_', expand=True)
        df['hours'] = time_split[0].astype(int)
        df['minutes'] = time_split[1].astype(int)
        df['seconds'] = time_split[2].str.strip().astype(int)
        df.drop(columns='hh_mm_ss', inplace=True)
        return df

    def create_weekday(self, df: pd.DataFrame) -> pd.DataFrame:
        """ 
        Adds a 'weekday' column to the DataFrame by calculating the day of the week based on the 'day' and 'month' columns.

        Parameters
        ----------
            df : The DataFrame that must contain at least 'month' and 'day' columns. 
                'month' is expected to have abbreviations like 'Mr', 'Apr', and 'Mai', 
                and 'day' contains the day of the month.

        Returns
        -------
            pd.DataFrame : The input DataFrame with the additional 'weekday' column, representing the weekday number (0 for Monday, 6 for Sunday).
        """

        df['month'] = df['month'].str.strip()
        month_map = {
            'Mr': 3, 'Apr': 4, 'Mai': 5
        }
        df['month'] = df['month'].map(month_map)
        df['year'] = 2017
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
        df['weekday'] = df['date'].dt.weekday
        return df

    def create_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """ 
        Creates the year, month, day, hours, minutes, seconds out of the timestamp.

        Parameters
        -------
            df : The DataFrame to modify.

        Returns
        -------
            pd.DataFrame: The modified DataFrame.
        """

        df['timestamp'] = pd.to_datetime(
            df[['year', 'month', 'day', 'hours', 'minutes', 'seconds']])
        return df

    def set_min_value(self, df: pd.DataFrame, V: float) -> pd.DataFrame:
        """ 
        Sets the minimum value of the first N columns in a DataFrame to V.

        Parameters
        -------
            df : The DataFrame to modify.
            N : The number of first columns to modify.
            V : The minimum value to set.

        Returns
        -------
            pd.DataFrame: The modified DataFrame.
        """

        condition = df < V
        return df.where(~condition, V)

    def get_cleaned_data_all_df(self, path: str, route_all: pd.DataFrame) -> pd.DataFrame:
        """
        This table contains the information of the different instances.
        -The first row contains the edge names, and three empty columns at the end.
        -Each of the subsequent row represents a different instance.
        -For each instance the weights of the edges can be found in their respective column. 
        -Additionally, the last three columns are the day of the month (1-31), the month 
        (Mr, Apr, or Mai), and the time stamp (hh_mm_ss), respectively.
        -The edges weight represent the speed that is estimated to be used in the edge, 
        as it is measured in mph.

        Parameters
        -------
            path : Path to data_all.
            route_all : The cleaned route_all DataFrame.

        Returns
        -------
            pd.DataFrame : The cleaned data_all DataFrame.
        """

        data_all = pd.read_csv(path)
        data_all = data_all.rename(columns={
            "Unnamed: 1045": "day", "Unnamed: 1046": "month", "Unnamed: 1047": "hh_mm_ss"})

        N = len(data_all.columns)-3
        data_all = self.convert_first_n_columns_to_float(data_all, N)
        data_all.iloc[:, :N] = self.set_min_value(data_all.iloc[:, :N], 3.0)

        data_all = self.split_times(data_all)
        data_all = self.create_weekday(data_all)
        data_all = self.create_timestamp(data_all)
        data_all_org = data_all.copy()
        int_columns = data_all.columns[:-9].astype(int)

        new_columns = list(int_columns) + list(data_all.columns[-9:])

        data_all.columns = new_columns

        desired_columns = list(range(1, 1309+1))
        existing_columns = set(data_all.columns[:-9].astype(int))
        missing_columns = [
            col for col in desired_columns if col not in existing_columns]

        missing_data = pd.DataFrame(
            24, index=data_all.index, columns=missing_columns)
        data_all = pd.concat([data_all, missing_data], axis=1)

        data_all = data_all[desired_columns]
        data_all = pd.concat([data_all, data_all_org.iloc[:, -9:]], axis=1)
        return data_all

    def get_cleaned_route_all_df(self, path: str) -> pd.DataFrame:
        """
        This file contains the route used in each instance. Each row represents an instance, 
        and contains the sequence of nodes that form the used route. Also, the last three columns are 
        the day of the month (1-31), the month (Mr, Apr, or Mai), and the time stamp (hh_mm_ss), respectively.
        The route to be analyzed starts in node 94 and ends in node 162.

        Parameters
        -------
            path : Path to route_all.

        Returns
        -------
            pd.DataFrame: The cleaned route_all DataFrame.
        """

        max_length = self.get_max_len(path)
        column_names = [i for i in range(max_length-3)]
        column_names += ["day", "month", "hh_mm_ss"]
        # next line of code might take up to 10 seconds to execute
        route_all = self.route_all_df_helper(
            path, column_names, max_length)

        route_all = self.split_times(route_all)
        route_all = self.create_weekday(route_all)
        route_all = self.create_timestamp(route_all)
        route_all.iloc[:, :-9] = route_all.iloc[:, :-9].astype(int)
        return route_all

    def get_cleaned_route_all_missing_last_day_df(self, path: str) -> pd.DataFrame:
        """
        This file contains the route used in each instance. Each row represents an instance, 
        and contains the sequence of nodes that form the used route. Also, the last three columns are 
        the day of the month (1-31), the month (Mr, Apr, or Mai), and the time stamp (hh_mm_ss), respectively.
        The route to be analyzed starts in node 94 and ends in node 162.

        Parameters
        -------
            path : Path to route_all.

        Returns
        -------
            pd.DataFrame: The cleaned route_all_missing_last_day DataFrame.
        """

        max_length = self.get_max_len(path)
        column_names = [i for i in range(max_length-3)]
        column_names += ["day", "month", "hh_mm_ss"]
        # next line of code might take up to 10 seconds to execute
        route_all_missing_last_day_df = self.route_all_df_helper(
            path, column_names, max_length)

        route_all_missing_last_day_df = self.split_times(
            route_all_missing_last_day_df)
        route_all_missing_last_day_df = self.create_weekday(
            route_all_missing_last_day_df)
        route_all_missing_last_day_df = self.create_timestamp(
            route_all_missing_last_day_df)

        route_all_missing_last_day_df.iloc[:, :-
                                           9] = route_all_missing_last_day_df.iloc[:, :-9].astype(int)

        return route_all_missing_last_day_df

    def get_cleaned_graph_df_for_plotting(self, path: str) -> pd.DataFrame:
        """
        This table consists on the following columns: 
        - Edge number
        - Edge name
        - Node A
        - Node B
        - Geographic X coordinate of node A 
        - Geographic Y coordinate of node A 
        - Geographic X coordinate of node B 
        - Geographic Y coordinate of node B

        Parameters
        -------
            path : Path to the graph DataFrame.

        Returns
        -------
            pd.DataFrame : The cleaned graph DataFrame.
        """

        graph = pd.read_csv(path)
        graph.columns = ['edge_number', 'edge_name', 'node_A', 'node_B',
                         'x_coordinate_A', 'y_coordinate_A', 'x_coordinate_B', 'y_coordinate_B']

        graph = graph.drop_duplicates(subset=['node_A', 'node_B'], keep=False)

        graph = graph.drop_duplicates(subset='edge_name')
        graph.index = pd.RangeIndex(len(graph))
        graph = graph.copy()
        graph.edge_number = pd.RangeIndex(1, len(graph)+1)

        return graph

    def get_cleaned_graph_df(self, path: str) -> pd.DataFrame:
        """
        This table consists on the following columns: 
        - Edge number
        - Edge name
        - Node A
        - Node B
        - Geographic X coordinate of node A 
        - Geographic Y coordinate of node A 
        - Geographic X coordinate of node B 
        - Geographic Y coordinate of node B

        Parameters
        -------
            path : Path to the graph DataFrame.

        Returns
        -------
            pd.DataFrame : The cleaned graph DataFrame.
        """

        graph = pd.read_csv(path)
        graph.columns = ['edge_number', 'edge_name', 'node_A', 'node_B',
                         'x_coordinate_A', 'y_coordinate_A', 'x_coordinate_B', 'y_coordinate_B']

        graph = graph.drop_duplicates(subset=['node_A', 'node_B'], keep=False)

        duplicated_edge_names = graph[graph.duplicated(
            'edge_name', keep=False)]['edge_name'].unique()

        for _, row in graph.iterrows():
            if row['edge_name'] in duplicated_edge_names:
                doubled_indexes = np.where(
                    graph['edge_name'] == row['edge_name'])[0]
                for doubled_index in doubled_indexes:
                    graph.iat[doubled_index,
                              4] = graph.iloc[doubled_indexes]['x_coordinate_A'].values.mean()
                    graph.iat[doubled_index,
                              5] = graph.iloc[doubled_indexes]['y_coordinate_A'].values.mean()
                    graph.iat[doubled_index,
                              6] = graph.iloc[doubled_indexes]['x_coordinate_B'].values.mean()
                    graph.iat[doubled_index,
                              7] = graph.iloc[doubled_indexes]['y_coordinate_B'].values.mean()

        graph = graph.drop_duplicates(subset='edge_name')
        graph.index = pd.RangeIndex(len(graph))
        graph = graph.copy()
        graph.edge_number = pd.RangeIndex(1, len(graph)+1)

        return graph
