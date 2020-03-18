"""Contains a class for retrieving and analysing data relating to Covid-19 pandemic"""
import os
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

URLS = {
    'deaths': 'https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv',
    'confirmed': 'https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv',
    'recovered': 'https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv',
    'worldometer': 'https://www.worldometers.info/coronavirus/',
}
PICKLE_FILE = 'output/data.pkl'
PICKLE_DIR = 'output'
def read_worldometer():
    """Returns a dataframe with the current Worldometer statistics"""
    response = requests.get(URLS['worldometer'])
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table', id='main_table_countries')
    data = []
    for row in table.find_all('tr'):
        cols = row.find_all('td')
        cols = [ele.text.strip() for ele in cols]
        data.append([ele for ele in cols if ele])
    return pd.DataFrame(
        data,
        columns=[
            'country',
            'total',
            'new',
            'deaths',
            'new_deaths',
            'recovered',
            'active',
            'serious',
            'cases_per_mil'])
        
def read_jhu_file(filename, label):
    """Read the Johns Hopkins University file format"""
    data = pd.read_csv(filename)
    return (
        data.
        rename(columns={
            'Province/State': 'province',
            'Country/Region': 'country',
            'Lat': 'lat',
            'Long': 'long'}).
        melt(id_vars=['country', 'province', 'lat', 'long']).
        rename(columns={'variable': 'date', 'value': label}).
        assign(date=lambda x: x.date.astype('datetime64'),
               province=lambda x: np.where(x.province.isnull(), x.country, x.province)).
        set_index(['country', 'province', 'date'])
    )

class Corona:
    """Retrieve, view and analyze data on the spread of Covid-19"""
    def __init__(self):
        self._data = None
        self._figsize = (14, 10)

    @property
    def data(self):
        """Contains the time series data for Covid-19 spread"""
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def figsize(self):
        """The size used for plots"""
        return self._figsize

    @figsize.setter
    def figsize(self, value):
        self._figsize = value

    def read_jhu(self, use_cache=True):
        "Reads data from Johns Hopkins University Github data. Updated daily."
        if use_cache and os.path.exists(PICKLE_FILE):
            self.data = pd.read_pickle(PICKLE_FILE)
            return

        results = {}
        for key in ['confirmed', 'recovered', 'deaths']:
            results[key] = read_jhu_file(URLS[key], key)
        self._data = (
            results['confirmed'].
            join(results['recovered'].drop(columns=['lat', 'long'])).
            join(results['deaths'].drop(columns=['lat', 'long'])).
            sort_index()
        )
        if os.path.exists(PICKLE_DIR):
            self.data.to_pickle(PICKLE_FILE)

    def do_plot(self, countries, columns, offsets=None, yscale='linear'):
        """Do a time series plot of the data
        Args:
            countries (list): List of strings with country names that should be
                plotted (see countries())
            columns (list): List of strings with columns to plot
                ('confirmed', 'deaths', 'recovered')

        Keyword Arguments:
            offsets (list): List of integers, same length as countries. For each country,
                move curve back so many days. If not supplied, all zeros is used. If shorter
                than countries, padded with zeros at the end.
            yscale (string): The scale of the y-axis ('linear', 'log', ...).
        """
        if not offsets:
            offsets = [0 for i in countries]
        if  len(offsets) < len(countries):
            offsets += [0 for i in range(len(countries) - len(offsets))]
        idx = pd.IndexSlice
        plt.figure(figsize=self.figsize)
        plt.yscale(yscale)
        for i, country in enumerate(countries):
            for col in columns:
                to_plot = (
                    self.data.
                    loc[idx[country, :, :], col].
                    groupby(level=[0, 2]).
                    sum().
                    shift(-offsets[i])
                )
                plt.plot(
                    to_plot.index.levels[1],
                    to_plot,
                    label=f"{country}({col})")
        plt.legend()

    def countries(self, regex=None):
        """Returns a list of countries available. Useful to search for countries if you don't
           know the exact name
        Args:
            regex (string): If provided, uses it to match case insensitively anywhere in
                country name
        """
        if regex:
            return [country for country in self.data.index.levels[0]
                    if re.search(regex, country, re.I)]
        return list(self.data.index.levels[0])

