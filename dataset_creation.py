# -----------------------------------------------------------------------------
# Project:     Bus Route Optimization
# File:        dataset_creation.py
# Company:     Stratton Oakmont
# Date:        September 13, 2024
# Version:     2.2.3
# Description: This script is used by the main.py file to create the distinct time groups.
# -----------------------------------------------------------------------------
# License:     Proprietary - Stratton Oakmont. All rights reserved.
# -----------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class DataSetCreator(object):
    def __init__(self) -> None:
        pass

    def create_no_rh_day(self, df: pd.DataFrame, no_rh_day_hours: list, no_rh_day_weekdays: list) -> pd.DataFrame:
        """
        Creates the DataFrames for the discrete time group: weekday midday.

        Parameters
        ----------
            df : The data_all DataFrame.
            no_rh_day_hours : Hours used for this discrete time group.
            no_rh_day_weekdays : Weekdays used for this discrete time group.

        Returns
        ----------
            pd.DataFrame : The masked DataFrame on the discrete time group.
        """

        no_rh_day = df[~df['hours'].isin(no_rh_day_hours)]
        no_rh_day = no_rh_day[~no_rh_day['weekday'].isin(
            no_rh_day_weekdays)]
        return no_rh_day

    def create_no_rh_night(self, df, no_rh_night_hours, no_rh_night_weekdays):
        """
        Creates the DataFrames for the discrete time group: weekday night.

        Parameters
        ----------
            df : The data_all DataFrame.
            no_rh_night_hours : Hours used for this discrete time group.
            no_rh_night_weekdays : Weekdays used for this discrete time group.

        Returns
        ----------
            pd.DataFrame : The masked DataFrame on the discrete time group.
        """

        no_rh_night = df[~df['hours'].isin(no_rh_night_hours)]
        no_rh_night = no_rh_night[~no_rh_night['weekday'].isin(
            no_rh_night_weekdays)]
        return no_rh_night

    def create_morning_rh(self, df, morning_rh_hours, morning_rh_weekdays):
        """
        Creates the DataFrames for the discrete time group: morning rush hour.

        Parameters
        ----------
            df : The data_all DataFrame.
            morning_rh_hours : Hours used for this discrete time group.
            morning_rh_weekdays : Weekdays used for this discrete time group.

        Returns
        ----------
            pd.DataFrame : The masked DataFrame on the discrete time group.
        """

        morning_rh = df[~df['hours'].isin(morning_rh_hours)]
        morning_rh = morning_rh[~morning_rh['weekday'].isin(
            morning_rh_weekdays)]
        return morning_rh

    def create_evening_rh(self, df, evening_rh_hours, evening_rh_weekdays):
        """
        Creates the DataFrames for the discrete time group: evening rush hour.

        Parameters
        ----------
            df : The data_all DataFrame.
            evening_rh_hours : Hours used for this discrete time group.
            evening_rh_weekdays : Weekdays used for this discrete time group.

        Returns
        ----------
            pd.DataFrame : The masked DataFrame on the discrete time group.
        """

        evening_rh = df[~df['hours'].isin(evening_rh_hours)]
        evening_rh = evening_rh[~evening_rh['weekday'].isin(
            evening_rh_weekdays)]
        return evening_rh

    def create_weekend_df(self, df, weekend_hours, weekend_weekdays):
        """
        Creates the DataFrames for the discrete time group: weekends.

        Parameters
        ----------
            df : The data_all DataFrame.
            weekend_hours : Hours used for this discrete time group.
            weekend_weekdays : Weekdays used for this discrete time group.

        Returns
        ----------
            pd.DataFrame : The masked DataFrame on the discrete time group.
        """

        weekend_df = df[~df['hours'].isin(weekend_hours)]
        weekend_df = weekend_df[~weekend_df['weekday'].isin(
            weekend_weekdays)]
        return weekend_df

    def seasonal_similarities_df(self) -> list[dict]:
        """
        Creates a list of dictonarys that conatin the hours and weekdays that are NOT in the discrete time group. 
        This allows for more efficient masking.

        Parameters
        ----------

        Returns
        ----------
            list[dict] : List of dictonarys with the hours and weekdays that are NOT in the discrete time group. 
        """

        seasonal_similarities = {}
        seasonal_similarities['no_rh_night'] = {}
        seasonal_similarities['no_rh_day'] = {}
        seasonal_similarities['morning_rh'] = {}
        seasonal_similarities['evening_rh'] = {}
        seasonal_similarities['weekend'] = {}

        for key in seasonal_similarities.keys():
            if 'night' in key:
                seasonal_similarities['no_rh_night']['hours'] = [
                    hour for hour in range(8, 18)]
                seasonal_similarities['no_rh_night']['weekdays'] = [
                    day for day in range(5, 7)]

            if 'day' in key:
                seasonal_similarities['no_rh_day']['hours'] = [
                    hour for hour in range(0, 10)] + [hour for hour in range(16, 24)]
                seasonal_similarities['no_rh_day']['weekdays'] = [
                    day for day in range(5, 7)]

            if 'morning' in key:
                seasonal_similarities['morning_rh']['hours'] = [
                    hour for hour in range(0, 8)] + [hour for hour in range(10, 24)]
                seasonal_similarities['morning_rh']['weekdays'] = [
                    day for day in range(5, 7)]

            if 'evening' in key:
                seasonal_similarities['evening_rh']['hours'] = [
                    hour for hour in range(0, 16)] + [hour for hour in range(18, 24)]
                seasonal_similarities['evening_rh']['weekdays'] = [
                    day for day in range(5, 7)]

            if 'weekend' in key:
                seasonal_similarities['weekend']['hours'] = []
                seasonal_similarities['weekend']['weekdays'] = [
                    day for day in range(0, 5)]

        return seasonal_similarities
