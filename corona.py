"""Contains a class for retrieving and analysing data relating to Covid-19 pandemic"""
import os
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from scipy import optimize
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
def expo(x, A, B):
    return A*np.exp(B * x)
def logistic(x, L, k, x0):
    return L/(1 + np.exp(-k*(x - x0)))

class Corona:
    """Retrieve, view and analyze data on the spread of Covid-19

    It uses data compiled by Johns Hopkins University and published on their Github repo"""
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
        plt.figure(figsize=self.figsize)
        plt.yscale(yscale)
        for i, country in enumerate(countries):
            for col in columns:
                to_plot = (
                    self.country_data(country).
                    loc[:, col].
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

    def country_data(self, country=None, offset_before_first_case=None):
        """Returns the data for a single country or the world
        Args:
            country (string): The country for which to fit, world if None (see countries())
        Keyword Arguments:
            offset_before_first_case (int): The number of days before the first case to start the
                dataset. If 0, dataset will start on day of first case. If None, all data is 
                returned.
        """
        idx = pd.IndexSlice
        if country:
            data = (
                self.data.
                loc[idx[country, :, :], :].
                groupby(level=[0, 2]).
                sum()
            )
        else:
            data = (
                self.data.
                groupby(level=2).
                sum()
            )
        if offset_before_first_case is None:
            return data

        first_non_zero = (data.confirmed > 0).values.argmax()
        data_start = max(first_non_zero - offset_before_first_case, 0)
        return data.iloc[data_start:, :]

    def curve_fit(self, country, end_range, offset_before_first_case=None, expo_day_limit=None):
        """Fits a exponential and logistic curve to the data for a given country
        Args:
            country (string): The country for which to fit (see countries())
            end_range (int): Where to end the projection of the fitted curves (number of days)
        Keyword Arguments:
            offset_before_first_case (int): See country_data()
            expo_day_limit (int): Only use this number of days to fit exponential function. 
                This is useful where the actual data has already turned logistic
        """
        data = self.country_data(country, offset_before_first_case).copy()
        x_values = np.arange(len(data.confirmed.values))
        y_values = data.confirmed.values
        
        if not expo_day_limit:
            p_expo, _ = optimize.curve_fit(expo, x_values, y_values, p0=[0.1, 0.1])
        else:
            p_expo, _ = optimize.curve_fit(expo, x_values[:expo_day_limit], y_values[:expo_day_limit], p0=[0.1, 0.1])
        try:
            p_logistic, _ = optimize.curve_fit(logistic, x_values, y_values,
                                               p0=[y_values[-1], 1, x_values[-1]])
        except RuntimeError:
            p_logistic = None

        x_fit = np.arange(end_range)
        y_fit1 = expo(x_fit, p_expo[0], p_expo[1])
        y_fit2 = None
        if p_logistic is not None:
            y_fit2 = logistic(x_fit, p_logistic[0], p_logistic[1], p_logistic[2])
            y_fit1 = np.where(y_fit1 > 2*y_fit2[-1], np.nan, y_fit1)

        top = y_fit1[~np.isnan(y_fit1)][-1]
        mid = end_range/2
        big_gap = 0.07 * top
        small_gap = 0.05 * top
        
        plt.figure(figsize=self.figsize)
        plt.scatter(x_values, y_values)
        plt.plot(x_fit, y_fit1, label="Exponential Fit")
        if p_logistic is not None:
            plt.plot(x_fit, y_fit2, label="Logistic Fit")
            log1 = round(p_logistic[0], 3)
            log2 = round(p_logistic[1], 3)
            log3 = round(p_logistic[2], 3)
            plt.text(mid, top-big_gap,
                     f"Logistic: $f(x) = \\frac{{{log1}}}{{1 + e^{{-{log2}(x - {log3})}}}}$",
                     fontsize=20, ha='center')
            plt.text(mid, top-2*big_gap, f"Logistic Midpoint: {log3}", ha='center')
            plt.text(mid, top-2*big_gap-small_gap, f"Logistic Maximum: {log1}", ha='center')
        plt.legend()
        
        exp1 = round(p_expo[0], 3)
        exp2 = round(p_expo[1], 3)
        text = plt.text(mid, top, f"Exponential: $f(x) = {exp1}e^{{{exp2}x}}$",
                        fontsize=20, ha='center')
        
        
        plt.title(country)

    def country_stats(self, country=None):
        """Provides some statistics for a given country or the world"""
        data = self.country_data(country, 0)
        return {
            "Number of Infections": data.confirmed[-1],
            "Number of Deaths": data.deaths[-1],
            "Number of Recoveries": data.recovered[-1],
            "Mortality Rate": data.deaths[-1]/data.confirmed[-1],
            "First Infection": data.shape[0],
        }
