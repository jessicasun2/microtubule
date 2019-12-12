import numpy as np
import pandas as pd

import bokeh_catplot

import bebi103

import bokeh.io
import bokeh.plotting
from bokeh.plotting import figure
bokeh.io.output_notebook()

import holoviews as hv
hv.extension('bokeh')

bebi103.hv.set_defaults()

import microtubule.general_utils as utils
import microtubule.distributions as dist

def two_ecdf(data1, data2):
    '''Returns a scatter plot of two ECDFs overlayed'''
    scatter1 = hv.Scatter(
        data = utils.ecdf_vals(data1),
        kdims = ['time to catastrophe (s)'],
        vdims = ['ECDF'],
        label = 'labeled'
    )

    scatter2 =  hv.Scatter(
        data = ecdf_vals(data2),
        kdims = ['time to catastrophe (s)'],
        vdims = ['ECDF'],
        label = 'not labeled'
    )

    chart = (scatter1 * scatter2).opts(legend_position='bottom_right')
    
    return chart

def sim_succ_poisson(process):
    p = bokeh_catplot.ecdf(
    data=pd.DataFrame({'time to catastrophe (1/β1)': process}),
    cats=None,
    val='time to catastrophe (1/β1)',
    title = 'ECDF of times to catastrophe',
    style='staircase'
    )

    return p

def cdf_vs_pdf(process, time, cdf):
    p = bokeh_catplot.ecdf(
        data=pd.DataFrame({'time to catastrophe (1/β1)': process}),
        cats=None,
        val='time to catastrophe (1/β1)',
        title = 'ECDF of times to catastrophe vs. Analytical CDF',
        style='staircase'
    )

    p.line(time, cdf, color='red')
    return p

def conf_ecdf(df):
    p = bokeh_catplot.ecdf(
        data=df,
        cats=['labeled'],
        val='time to catastrophe (s)',
        ptiles=[2.5, 97.5],
        conf_int = True,
        style='staircase'
    )

    p.legend.location = 'bottom_right'
    
    return p

def plot_DKW_inequal(data):
    p = bokeh_catplot.ecdf(
    data=pd.DataFrame({'time to catastrophe (s)': data}),
    cats=None,
    val='time to catastrophe (s)',
    conf_int = True,
    title='Computed 95% confidence interval vs. DKW inequality',
    x_axis_label='time to catastrophe (s)',
    y_axis_label = 'ECDF',
    style='staircase'
    )

    # actual distribution 
    p.circle(labeled, labeled_lower, color='red', legend_label='lower bound')
    p.circle(labeled, labeled_upper, color='blue', legend_label='upper bound')

    p.legend.location = 'bottom_right'
    
    return p



