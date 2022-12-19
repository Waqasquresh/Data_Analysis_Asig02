
# import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# load the the data of international soccer team
# into pandas dataframe
soccer_data = pd.read_csv('spi_matches.csv')


# we will make a dataframe which will contain data
# related to UEFA Champions League only because
# now in our soccer_data dataframe we contain data related to all the leagues
uefa_data = soccer_data[soccer_data['league'] == 'UEFA Champions League']


# print all the teams playing in UEFA Champions League
print(uefa_data.team1.unique())


# we will filter data related to specific teams
# we will take data of 8 teams and do analysis for them
def six_team_data(name, df):
    ''' This function take in list of club teams and the dataframe that team in
        as input and return a dataframe having data of all the teams.
    '''
    team1 = df[df['team1'] == name[0]]
    team2 = df[df['team1'] == name[1]]
    team3 = df[df['team1'] == name[2]]
    team4 = df[df['team1'] == name[3]]
    team5 = df[df['team1'] == name[4]]
    team6 = df[df['team1'] == name[5]]
    team7 = df[df['team1'] == name[6]]
    team8 = df[df['team1'] == name[7]]
    return pd.concat([team1, team2, team3, team4, team5, team6, team7, team8])


# making list of 8 teams of which data we want to analysis
team_list = ['Barcelona', 'Paris Saint-Germain', 'Manchester City',
             'Juventus', 'Real Madrid', 'Arsenal',
             'Manchester United', 'Liverpool']

# passing the team_list list to six_team_data as arg
# which will return a data/records related to the teams
# which we choose in team_list
teams_df = six_team_data(team_list, uefa_data)


# we want to analysis the probabilty of home winning over
barcelona_data = teams_df[teams_df['team1'] == 'Barcelona'].dropna()
psg_data = teams_df[teams_df['team1'] == 'Paris Saint-Germain'].dropna()
mc_data = teams_df[teams_df['team1'] == 'Manchester City'].dropna()
juv_data = teams_df[teams_df['team1'] == 'Juventus'].dropna()
rm_data = teams_df[teams_df['team1'] == 'Real Madrid'].dropna()
arsenal_data = teams_df[teams_df['team1'] == 'Arsenal'].dropna()
mu_data = teams_df[teams_df['team1'] == 'Manchester United'].dropna()
liverpool_data = teams_df[teams_df['team1'] == 'Liverpool'].dropna()


# function which will plot line graph for the selected teams
def plot_line_graph():
    '''
    This function plot the line plot upon calling 
    the line plot shows probability of winning of 
    home teams over the seasons for all home teams

    '''
    # set fig size
    plt.figure(figsize=(15, 10))
    # set the line plot value on x-axis and y-axis
    plt.plot(barcelona_data.season, barcelona_data.prob1,
             '--b', label='Barcelona')
    plt.plot(psg_data.season, psg_data.prob1,
             '--', label='Paris Saint-Germain')
    plt.plot(mc_data.season, mc_data.prob1, '-', label='Manchester City')
    plt.plot(juv_data.season, juv_data.prob1, '-', label='Juventus')
    plt.plot(rm_data.season, rm_data.prob1, '--', label='Real Madrid')
    plt.plot(arsenal_data.season, arsenal_data.prob1, '--c', label='Arsenal')
    plt.plot(mu_data.season, mu_data.prob1, '-', label='Manchester United')
    plt.plot(liverpool_data.season, liverpool_data.prob1,
             '--', label='Liverpool')

    # Set the X-axis label and make it bold
    plt.xlabel('Season', fontweight='bold')

    # set the title
    plt.title("Probability of the home team winning")

    # show the legends on the plot and place it on suitable position
    plt.legend(bbox_to_anchor=(0.18, 0.3), shadow=True)

    # show the line plot
    plt.show()


plot_line_graph()


# lets see barcelona soccer power index over the seasons from 2016
# to 2022 if it increased or decrease
spi_data = teams_df.groupby(['season', 'team1'])['spi1'].mean()
spi_data = spi_data.unstack(0)


# using bar plot we can easily analyize if the spi increased or decreased
def plot_bar_plot():
    '''
    This function plot bar chart for all teams in dataframe
    it shows soccer power index of home teams between 2016 and
    2022
    '''

    # set fig size
    plt.figure(figsize=(15, 10))

    # set width of bars
    barWidth = 0.1

    # plot bar charts
    plt.bar(np.arange(spi_data.shape[0])+0.1, spi_data[2016],
            color='blue', width=barWidth, label='2016')
    plt.bar(np.arange(spi_data.shape[0])+0.2, spi_data[2017],
            color='yellow', width=barWidth, label='2017')
    plt.bar(np.arange(spi_data.shape[0])+0.3, spi_data[2018],
            color='red', width=barWidth, label='2018')
    plt.bar(np.arange(spi_data.shape[0])+0.4, spi_data[2019],
            color='darkviolet', width=barWidth, label='2019')
    plt.bar(np.arange(spi_data.shape[0])+0.5, spi_data[2020],
            color='peru', width=barWidth, label='2020')
    plt.bar(np.arange(spi_data.shape[0])+0.6, spi_data[2021],
            color='slategray', width=barWidth, label='2021')
    plt.bar(np.arange(spi_data.shape[0])+0.7, spi_data[2022],
            color='lime', width=barWidth, label='2022')

    # show the legends on the plot
    plt.legend()

    # set the x-axis label
    plt.xlabel('Clubs', fontsize=15)

    # add title to the plot
    plt.title("Soccer Power Index", fontsize=15)

    # add countries names to the 11 groups on the x-axis
    plt.xticks(np.arange(spi_data.shape[0])+0.2,
               ('Arsenal', 'Barcelona', 'Juventus', 'Liverpool', 'Manchester City',
                'Manchester United', 'Paris Saint-Germain', 'Real Madrid'),
               fontsize=10, rotation=45)

    # show the plot
    plt.show()


# show the plot graph
plot_bar_plot()

# filtering the data keeping all the rows and selected specific columns
juv = juv_data.iloc[:, 6:19]


def correlation_matrix():
    '''
    This function plot the correlation heatmap for Juventus club
    showing its feature interconnection relation
    '''
    # create correlation matrix
    corr_matrix = juv.corr()

    plt.figure(figsize=(10, 5))

    # using seaborn library to create heatmap
    sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu")

    plt.title("Correlation Heatmap of Juventus")

    plt.show()


correlation_matrix()
