import pandas as pd
import numpy as np

from datetime import timedelta

import geopandas as gpd

import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec
from matplotlib.colors import ListedColormap
from scipy import stats

from .geolocations import *

COLORS = {"light_orange":"#E69F00",
             "light_blue":"#56B4E9",
             "teal":"#009E73",
             "yellow":"#F0E442",
             "dark_blue":"#0072B2",
             "dark_orange":"#D55E00",
             "pink":"#CC79A7",
             "purple":"#9370DB",
             "black":"#000000",
             "silver":"#DCDCDC"}

COLOR_MAP = {
        0:"#e7e1ef",
        1:"#c994c7",
        2:"#dd1c77"
    }

START_DATE = pd.to_datetime("2021-04-19")
US_POPULATION = 329500000

def load_national_trend_data():
    """ Returns dataframe of national vaccine trends.
    """
    df = pd.read_csv("../data/national_level_vaccination_trends.csv")
    df = df[df["Date Type"] == "Admin"]
    df["Date"] = [pd.to_datetime(d) for d in df["Date"]]
    df.set_index("Date", drop = True, inplace = True)
    return df

def load_county_trend_data(county, state, download_data= False):
    """ Returns dataframe of county level vaccine trends.

    Inputs: 
        county - (str) Capitalized county name "<Name> County"
        state - (str) Upper-case two letter state abbreviation.
        download_data: (bool) Download full dataset if True

    Returns:
        Returns county and state level vaccination data for 
        county, state, if download_data = True then it is 
        downloaded directly from the CDC website: 
        https://data.cdc.gov/Vaccinations/COVID-19-Vaccinations-
        in-the-United-States-County/8xkx-amqh.

    """
    c = county.lower()
    s = state.lower()
    
    if download_data == True:
        # Caution: this will take a long time.
        df = pd.read_csv(
        "https://data.cdc.gov/api/views/8xkx-amqh/rows.csv?accessType=DOWNLOAD")
        df = df[(df["Recip_County"] == "{} County".format(county.capitalize())
            ) & (df["Recip_State"] == state.upper())]

    else:
        df = pd.read_csv("../data/{}_county_{}_vaccination_trends.csv".format(
            c,s), index_col = 0)
    
    df["Date"] = [pd.to_datetime(d) for d in df["Date"]]
    df.sort_values(by = "Date", inplace = True)
    df.set_index("Date", drop = True, inplace = True)
    
    return df

def national_complete_pct(df):
    """ Returns timeseries of percentage completely vaccinated.

    Input: 
        df - (dataframe) national vaccination trends.

    Output: 
        Dataframe of national percentage fully vaccinated by date.

    """
    complete = 100 * df['People Fully Vaccinated Cumulative']/US_POPULATION
    return complete.loc[START_DATE:]

def national_one_dose_pct(df):
    """ Returns timeseries of percentage with one dose. 

    Input: 
        df - (dataframe) national vaccination trends.

    Output:
        Dataframe of national percentage with one dose by date.
    """
    one_dose = 100 * df[
            'People Receiving 1 or More Doses Cumulative']/US_POPULATION
    return one_dose.loc[START_DATE:]

def national_expected_complete_pct(df):
    """ Returns timeseries of expected percentage completely vaccinated.

    Input: 
        df - (dataframe) national vaccination trends.

    Output: 
        Dataframe of national percentage expected complete by date.
    """
    one_dose = 100 * df['People Receiving 1 or More Doses Cumulative'
                            ]/US_POPULATION     
    expected = one_dose.loc[START_DATE - timedelta(days = 42
        ):one_dose.index[-1] - timedelta(days = 42)]
    expected = pd.Series(expected.values, index = pd.date_range(
        START_DATE, one_dose.index[-1]))
    return expected

def county_complete_pct(df):
    """ Returns timeseries of percentage completely vaccinated.

    Input: 
        df - (dataframe) county level vaccination trends.

    Output: 
        Dataframe of county level percentage complete by date.
    """
    return df["Series_Complete_Pop_Pct"].loc[START_DATE:,]

def county_one_dose_pct(df):
    """ Returns timeseries of percentage with one dose.
    
    Input: 
        df - (dataframe) county level vaccination trends.

    Output: 
        Dataframe of county level percentage with one dose by date.
    """
    return df['Administered_Dose1_Pop_Pct'].loc[START_DATE:,]

def county_expected_complete_pct(df):
    """ Returns timeseries of percentage expected completely vaccinated.
    
    Input: 
        df - (dataframe) county level vaccination trends.

    Output: 
        Dataframe of county level percentage expected complete by date.
    """
    one_dose = df['Administered_Dose1_Pop_Pct']
    expected = one_dose.loc[START_DATE - timedelta(days = 42
        ):one_dose.index[-1] - timedelta(days = 42)]
    expected = pd.Series(expected.values, index = pd.date_range(
        START_DATE, one_dose.index[-1]))
    
    return expected

def vaccine_trends_plot(county = None, 
                        state = None,
                        show_us_current = False,
                        download_data = False):
    """ Returns line plot of county vaccine trends.
    
    Inputs: 
        county - (str) Capitalized county name or None for national 
            level data.
        state - (str) Upper-case two letter state abbreviation or None
            for national level data.
        show_us_current - (bool) set to False to hide vertical line
            at current us vaccination rate.
        download_data - (bool) set to True to download data directly 
            from CDC webiste.  Warning: this is slow.

    Returns: 
        Line plot of percentage complete, one-dose, and expected complete
        over time with optional vertical line at national current level.
    """
    df = load_national_trend_data()
    
    complete = national_complete_pct(df)
    one_dose = national_one_dose_pct(df)
    expected = national_expected_complete_pct(df)
    
    fig, ax = plt.subplots(figsize = (8,5))
    
    # Add horizontal line at current completely vaccinated.
    if show_us_current == True:
        y = df.index[-1].year
        m = f"{df.index[-1].month:02}"
        d = f"{df.index[-1].day:02}"
        ax.plot((one_dose.index[0], one_dose.index[-1]),
                (complete.iloc[-1],complete.iloc[-1]), 
                color = "k", 
                linewidth = 1, 
                linestyle = "--", 
                zorder = 0,
                label = "US Complete Vaccination Rate {}-{}-{}".format(y,m,d))
        ax.annotate("{}%".format(np.around(complete.iloc[-1], decimals = 1)), 
                    (one_dose.index[0], complete.iloc[-1]+1))

    ax.set_title("US National Vaccination Rates", fontsize = 15) 

    # Load county data.
    if county:

        if state:

            c = county.lower().split(" county")[0]
            s = state.upper()
            df = load_county_trend_data(county = c, state = s, 
                download_data = download_data)
        
            complete = county_complete_pct(df)
            one_dose = county_one_dose_pct(df)
            expected = county_expected_complete_pct(df)

            ax.set_title("Vaccination Rates in {} County, {}".format(
                                c.capitalize(), s), fontsize = 15)   

        else:
            raise ValueError("A two-letter state abbreviation must be given.")
    
    # Plot trends
    ax.plot(one_dose, 
            color = COLORS["dark_blue"], 
            linewidth = 3, 
            label = "One Dose")
    ax.plot(complete, 
            color = COLORS["light_orange"], 
            linewidth = 3, 
            label = "Completely Vaccinated")
    ax.plot(expected, 
            color = "gray", 
            linestyle = "dotted",
            linewidth = 3, 
            label = "Expected Completely Vaccinated", 
            zorder = 0)


    ax.set_xlabel("Date", fontsize = 12)
    ax.set_yticks([0,20,40,60,80])
    ax.set_ylim(0,90)
    ax.set_yticklabels(["{}%".format(20*i) for i in range(5)])
    ax.set_ylabel("Percentage", fontsize = 12)
    ax.legend(loc = "lower right", prop = {"size":12})
    plt.show()
    return None

def relative_vaccine_trends_plot(county = None,
                                state = None,
                                download_data = False):
    """ Returns bar chart of percentage +/- expected complete.

    Inputs: 
        county - (str) Capitalized county name or None for national 
            level data.
        state - (str) Upper-case two letter state abbreviation or None
            for national level data.
        download_data - (bool) set to True to download data directly 
            from CDC webiste.  Warning: this is slow.

    Returns: 
        Bar chart showing percentage points above of below the 
        expected vaccine rate as a function of time.
    """
    df = load_national_trend_data()
    complete = national_complete_pct(df)
    expected = national_expected_complete_pct(df)

    fig, ax = plt.subplots(figsize = (8,5))
    ax.set_title("Relative US National Vaccination Rates", fontsize = 15)

    if county:

        if state:
            c = county.lower().split(" county")[0]
            s = state.upper()
            df = load_county_trend_data(county = c, state = s, 
                download_data = download_data)
        
            complete = county_complete_pct(df)
            one_dose = county_one_dose_pct(df)
            expected = county_expected_complete_pct(df)

            ax.set_title("Relative Vaccination Rates in {} County, {}".format(
                                c.capitalize(), s), fontsize = 15)  

        else:
            raise ValueError("A two-letter state abbreviation must be given.")
    

    # Compute difference between expected and actual.
    diff = complete - expected
    diff_weekly = pd.DataFrame(index = pd.date_range(diff.index[0],
                                                diff.index[-1], freq = "W"),
                             columns = ["mean"])

    for i in range(diff.shape[0]-1):
        start = diff.index[i]
        end = diff.index[i+1]
        diff_weekly.loc[start,'mean'] = diff.loc[start:end].mean()

    color = [COLORS["teal"] if t >= 0 else COLORS["pink"] for t in 
                                diff_weekly["mean"]]
    
    # Plot trends.
    ax.bar(x = diff_weekly.index, width = 1, 
        height = diff_weekly["mean"],
        color = color)
    
    # Add empty plot to generate legend.
    ax.plot([],[],
        color = COLORS["teal"], 
        label= "More people than expected are completely vaccinated", 
        linewidth = 3)
    ax.plot([],[],
        color = COLORS["pink"], 
        label= "Fewer people than expected are completely vaccinated", 
        linewidth = 3)
    
    ax.set_ylabel("Percentage Points", fontsize = 12)
    ax.set_xlabel("Date", fontsize = 12)
    ax.set_ylim(-12,12)
    ax.legend(loc = "lower left", prop = {"size":12})
    
    plt.show()
    return None

def plot_triangulated_county(geo_df, bounding_box = None, restricted = False, aspect_ratio = 1):
    """ Plots county with triangular regions.
    
    Inputs: 
        geo_df: (dataframe) geographic datatframe including county geometry.
        bounding_box: (list) list of 4 vertices determining a bounding box 
                where agents are to be added.  If no box is given, then the 
                bounding box is taken as the envelope of the county.
        restricted: (bool) if True then region is restrict to bounding box.
        aspect_ratio: (float) aspect ratio of final plot.

    Returns: 
        Boundary of county and triangulation of region.
    """
    tri_dict = make_triangulation(geo_df)
    tri_df = gpd.GeoDataFrame({"geometry":[Polygon(t) for t in tri_dict["geometry"]["coordinates"]]})
    
    # Establish initial CRS
    tri_df.crs = "EPSG:3857"

    # Set CRS to lat/lon
    tri_df = tri_df.to_crs(epsg=4326) 

    fig, ax = plt.subplots(figsize = (10,10))
    linewidth = 1
    # Get bounding box geometry.
    if bounding_box is not None:
        sq_df = gpd.GeoDataFrame({"geometry":[Polygon(bounding_box)]})
        
        # Get bounded triangles.
        if restricted == True:
            inset = [i for i in tri_df.index if tri_df.loc[i,"geometry"].within(sq_df.loc[0,"geometry"])]
            tri_df = tri_df.loc[inset,:].copy()

            # Set plot limits.
            minx = np.array(bounding_box)[:,0].min()
            miny = np.array(bounding_box)[:,1].min()
            maxx = np.array(bounding_box)[:,0].max()
            maxy = np.array(bounding_box)[:,1].max()
            
            ax.set_xlim(minx - .0005,maxx + .0005)
            ax.set_ylim(miny - .0005,maxy + .0005)

            linewidth = 4

    # Plot triangles
    tri_df.boundary.plot(ax = ax, 
                        alpha=1, 
                        linewidth = linewidth,
                        edgecolor = COLORS["light_blue"])

    # Plot county boundary.
    geo_df.crs = "EPSG:3857"
    geo_df = geo_df.to_crs(epsg=4326) 
    geo_df.boundary.plot(ax = ax,edgecolor = "black", linewidth = linewidth)

    # Plot bounding box.
    if bounding_box is not None:
        if restricted == False:
            sq_df.boundary.plot(ax = ax, 
                                alpha = 1, 
                                linestyle = "--", 
                                linewidth = 2, 
                                color = COLORS["dark_orange"])

    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)
    plt.show()

    return None

def plot_agents_on_triangle(triangle_object, agent_df):
    """ Returns triangle filled with agents.

    Inputs: 
        triangle_object : (polygon) shapely triangle object.
        agent_df: (dataframe) x,y coordinates for agents.

    Outputs: 
        Plot of points on triangle.
    """
    fig, ax = plt.subplots(figsize = (8,8))
    df = gpd.GeoDataFrame({"geometry": triangle_object}, index = [0])
    df.boundary.plot(ax = ax, alpha=1, edgecolor = COLORS["light_blue"])
    ax.scatter(agent_df["x"], agent_df["y"], color = COLORS["dark_blue"], 
        zorder = 0)
    ax.set_axis_off()
    plt.show()
    return None

def plot_agents_with_belief_and_weight(belief_df):
    """ Returns triangle filled with agents.

    Inputs: 
        triangle_object : (polygon) shapely triangle object.
        agents: (dataframe) x,y coordinates for agents.

    Outputs: 
        Plot of points on triangle.
    """
    fig, ax = plt.subplots(figsize = (8,8))
    ax.scatter(belief_df["x"], belief_df["y"],
        s = [int(w) for w in belief_df["weight"].values],
        c = [COLOR_MAP[b] for b in belief_df["belief"].values])

    ax.scatter([],[],color = COLOR_MAP[0], label = "Not Hesitant")
    ax.scatter([],[],color = COLOR_MAP[1], label = "Hesitant or Unsure")
    ax.scatter([],[],color = COLOR_MAP[2], label = "Strongly Hesitant")
    ax.set_axis_off()     
    plt.legend(loc = "best", prop={'size': 15})
    
    plt.show()
    return None


def plot_network(model):
    """ OpinionNetworkModel instance
    """
    # Load point df.
    belief_df = model.belief_df

    fig, ax = plt.subplots(figsize = (8,8))
    
    op = model.include_opinion
    wt = model.include_weight

    adjacency_df = model.adjacency_df
    adjacency_df.columns = [int(i) for i in adjacency_df.columns]

    cc = model.clustering_coefficient
    md = model.mean_degree
    # Plot people.
    ax.scatter(belief_df["x"], belief_df["y"], 
               s = [int(w) for w in belief_df["weight"].values],
               c = [COLOR_MAP[b] for b in belief_df["belief"].values])

    for j in belief_df.index:
        for k in np.where(adjacency_df.loc[j,:] == 1)[0]:
            ax.plot((belief_df.loc[j,"x"], belief_df.loc[k,"x"]),
                            (belief_df.loc[j,"y"], belief_df.loc[k,"y"]),                            
                            color = "k", lw = .5, zorder = 0)

    # Turn off axes
    ax.set_axis_off()
    
    # Add title
    title = "Connections based on physical distance"
    if op == True:
        if wt == True:
            title = "Connections based on physical distance, opinion proximity and weight"
        if wt == False:
            title = "Connections based on physical distance and opinion proximity"
    if op == False:
        if wt == True:
            title = "Connections based on physical distance and weight"
        if wt == False:
            title = "Connections based on physical distance"



    title = title + "\n clustering coefficient: " + str(
        np.around(cc, decimals = 3)) + "\n average in-degree: " + str(
        np.around(md, decimals = 1))
    ax.set_title(title)
        
    # Add legend
    ax.scatter([],[],color = COLOR_MAP[0], label = "Not Hesitant")
    ax.scatter([],[],color = COLOR_MAP[1], label = "Hesitant or Unsure")
    ax.scatter([],[],color = COLOR_MAP[2], label = "Strongly Hesitant")     
    plt.legend(loc = "best")
    plt.axis()

    plt.show()

    None

def plot_opinion_network(network_info):
    """ Draws the opinion network at a specific time.

    Inputs: 
        network_info : (dictionary) all the info of the network.

    Outputs: 
        Plots the opinion network.
    """
    # Load point df.
    belief_df = network_info['belief_df']

    fig, ax = plt.subplots(figsize = (8,8))

    adjacency_df = network_info['adjacency_df']
    adjacency_df.columns = [int(i) for i in adjacency_df.columns]

    cc = network_info['clust_coeff']
    md = network_info['mean_degree']
    
    N = 256
    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(231/N, 221/N, N)
    vals[:, 1] = np.linspace(225/N, 28/N, N)
    vals[:, 2] = np.linspace(239/N, 119/N, N)
    cmap = ListedColormap(vals)
    
    # Plot people.    
    ax.scatter(belief_df["x"], belief_df["y"], 
               s = [int(w) for w in belief_df["weight"].values],
               c = belief_df["belief"].values,
               cmap = cmap)

    for j in belief_df.index:
        for k in np.where(adjacency_df.loc[j,:] == 1)[0]:
            ax.plot((belief_df.loc[j,"x"], belief_df.loc[k,"x"]),
                            (belief_df.loc[j,"y"], belief_df.loc[k,"y"]),                            
                            color = "k", lw = .5, zorder = 0)

    # Turn off axes
    ax.set_axis_off()
    
    # Add title
    title = "Connections"

    title = title + "\n clustering coefficient: " + str(
        np.around(cc, decimals = 3)) + "\n average in-degree: " + str(
        np.around(md, decimals = 1))
    ax.set_title(title)
        
    # Add legend
    ax.scatter([],[],color = COLOR_MAP[0], label = "Not Hesitant")
    ax.scatter([],[],color = COLOR_MAP[1], label = "Hesitant or Unsure")
    ax.scatter([],[],color = COLOR_MAP[2], label = "Strongly Hesitant")     
    plt.legend(loc = "best")
    plt.axis()

    plt.show()

    None
    
    
def get_ridge_plot(dynamic_belief_df, 
                phases = [], 
                reach_dict = None, 
                show_subplot_labels = True,
                show_title = True):
    """ Ridgeplot of updating beliefs.

    Inputs: 
        dynamic_belief_df: (dataframe) updating beliefs across multiple phases
        phases: (list) phases to show in plot.
        reach_dict: (dictionary) value is propotional reach of key.
        show_subplot_labels: (bool) if True show subplot labels.
        show_title: (bool) if True show plot title.

    Ouputs: 
        Ridgeplot of updating belief distributions over phases.
    """
    if phases == []:
        phases = dynamic_belief_df.shape[1] - 1
        if phases < 5:
            phases = [t for t in range(phases +1)]
        else:
            t = phases // 5
            phases = [0] + [t * (i+1) for i in range(1,5)]

    c = ['#edf8fb','#b2e2e2','#66c2a4','#2ca25f','#006d2c']
    xx = np.linspace(0, 2, 1000)

    gs = grid_spec.GridSpec(len(phases),1)
    fig = plt.figure(figsize=(8,4))

    i = 0

    ax_objs = []

    for p in range(len(phases)):
        ax_objs.append(fig.add_subplot(gs[i:i+1, 0:]))
        x = dynamic_belief_df[phases[p]].values
        kde = stats.gaussian_kde(x)
        ax_objs[-1].plot(xx, kde(xx), color = c[0])
        ax_objs[-1].fill_between(xx,kde(xx), color=c[p], alpha = 0.8)

        ax_objs[-1].set_yticks([])
        ax_objs[-1].set_yticklabels([])
        ax_objs[-1].set_ylabel('')
        
        #ax_objs[-1].set_axis_off()
        if show_subplot_labels == True:
            ax_objs[-1].text(2.1,0,"{} time steps".format(phases[p]),
                fontweight = "bold",
                fontsize=10,ha="left")

        # make background transparent
        rect = ax_objs[-1].patch
        rect.set_alpha(0)

        if i == len(phases)-1:
            ax_objs[-1].set_xticks([0,1,2])
            ax_objs[-1].set_xticklabels([0, 
                                        1,
                                        2])
        else:
            ax_objs[-1].set_xticks([])
            ax_objs[-1].set_xticklabels([])

        spines = ["top","right","left","bottom"]
        for s in spines:
            ax_objs[-1].spines[s].set_visible(False)
            
        i += 1
        
    
    gs.update(hspace=-0.7)
    left = int(reach_dict[0] * 100)
    right = int(reach_dict[2] *100)
    if show_title == True:
        plt.title("Left Reach: {}%    Right Reach: {}%".format(left, right), 
            y=-.4, fontweight = "bold")

    return None

def plot_mean_beliefs(results):
    """ Line graph of the mean beliefs of the model at every time step

    Inputs: 
        results: (dataframe) contains all the results of a simulation
                 must be explicitly saved when running a simulatoin

    Ouputs: 
        Line graph of the mean beliefs.
    """
    means = []
    for result in results:
        means.append(result['belief_df']['belief'].mean(axis = 0))
    time_steps = range(len(means))
    y_interval = np.arange(0,2)
    fig, ax = plt.subplots()
    ax.set_title('Average belief across timesteps', fontsize=14)
    ax.set_xlabel('Time step', fontsize=14)
    ax.set_ylabel('Mean belief', fontsize=14)
    ax.plot(time_steps,means, color = 'red', marker = 'o')
    ax.set_xlim(0, max(time_steps))
    ax.set_ylim(0, 2)
    ax.grid(axis = 'y', visible = True)
    plt.show
    
def plot_mean_degree(results):
    """ Line graph of the mean degree of the model at every time step

    Inputs: 
        results: (dataframe) contains all the results of a simulation
                 must be explicitly saved when running a simulatoin

    Ouputs: 
        Line graph of the mean beliefs.
    """
    means = []
    for result in results:
        means.append(result['mean_degree'])
    time_steps = range(len(means))
    y_interval = np.arange(0,2)
    fig, ax = plt.subplots()
    ax.set_title('Average in-degree across timesteps', fontsize=14)
    ax.set_xlabel('Time step', fontsize=14)
    ax.set_ylabel('average in-degree', fontsize=14)
    ax.plot(time_steps,means, color = 'blue', marker = 'o')
    ax.set_xlim(0, max(time_steps))
    ax.grid(axis = 'y', visible = True)
    plt.show
