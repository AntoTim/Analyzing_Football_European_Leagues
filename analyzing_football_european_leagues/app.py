#################################################################################

# https://analyzingfootballeuropeanleagues.streamlit.app
# Copyright (c) 2024 Anto Tim

#################################################################################





import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler 
from scipy.stats import zscore





###################################################################################################

###################################################################################################

###################################################################################################

# Load up the data

df_u = pd.read_csv('analyzing_football_european_leagues/understat.com.csv')
#df_u = pd.read_csv('understat.com.csv')
df_p = pd.read_csv('analyzing_football_european_leagues/understat_per_game.csv')
#df_p = pd.read_csv('understat_per_game.csv')


df_u.rename(columns={'Unnamed: 0': 'League'}, inplace=True)
df_u.rename(columns={'Unnamed: 1': 'Year'}, inplace=True)
df_p['date'] = pd.to_datetime(df_p['date'])

league_names_mapping = {
    'EPL': 'Premier League',
    'La_liga': 'La Liga',
    'Bundesliga': 'Bundesliga',
    'Ligue_1': 'Ligue 1',
    'Serie_A': 'Serie A',
    'RFPL': 'Russian League'
}

###################################################################################################

###################################################################################################

###################################################################################################

# Functions for League Table / Season Progress Chart / Matchweek Analysis 

# Function to display the final standings
def plot_league_table_by_year_and_league(year, league):

    df_filtered = df_u[(df_u['Year'] == year) & (df_u['League'] == league)]
    
    if df_filtered.empty:
        st.write(f"No data found for the year {year} and league {league}.")
        return
    
    columns_to_display = ['position', 'team', 'matches', 'wins', 'draws', 'loses', 'scored', 'missed', 'pts']
    df_filtered = df_filtered[columns_to_display]
    
    df_filtered = df_filtered.sort_values(by='position').reset_index(drop=True)

    styled_df = df_filtered.style.applymap(
        lambda _: "background-color: Green;", subset=([0], slice(None))
    ).applymap(
        lambda _: "background-color: LightGreen;", subset=([1, 2, 3, 4], slice(None))
    ).applymap(
        lambda _: "background-color: #FFA756;", subset=([5], slice(None))
    ).applymap(
        lambda _: "background-color: #FF9999;", subset=([6, 7, 8], slice(None))
    )

    def highlight_last_three_rows(s):
        if s.name in df_filtered.index[-3:]:  
            return ['background-color: red'] * len(s)  
        return [''] * len(s) 

    styled_df = styled_df.apply(highlight_last_three_rows, axis=1)
    st.dataframe(styled_df)

def plot_final_standings_and_trend(year, league):

    df_filtered = df_p[(df_p['year'] == year) & (df_p['league'] == league)]
    if df_filtered.empty:
        st.warning(f"No data found for the year {year} and league {league}.")
        return    
    
    final_standings = df_filtered.groupby('team').agg({
        'pts': 'sum'
    }).reset_index()
    
    final_standings['position'] = final_standings['pts'].rank(ascending=False, method='min')
    
    final_standings = final_standings.sort_values(by='position')

    team_order = final_standings['team'].tolist()

    df_filtered['cumulative_pts'] = df_filtered.groupby('team')['pts'].cumsum()

    fig = px.line(
        df_filtered,
        x='date',
        y='cumulative_pts',
        color='team',
        line_group='team',
        title=f"Trend of {league} standings in {year}-{year + 1}",
        labels={"cumulative_pts": "Total Points", "date": "Date"},
        color_discrete_sequence=px.colors.qualitative.Dark2
    )
    
    fig.update_layout(
        title=f"Trend of {league} standings in {year}-{year + 1}",
        xaxis_title="Date",
        yaxis_title="Total Points",
        xaxis_tickangle=-45,
        legend_title_text='Team'
    )

    st.plotly_chart(fig)

# Display the number of goals per matchweek
def plot_goals_per_matchweek(year, league, match_type='Overall'):

    df_filtered = df_p[(df_p['year'] == year) & (df_p['league'] == league)]

    df_filtered['matchweek'] = df_filtered.groupby('team').cumcount() + 1

    if match_type == 'Home':
        df_filtered = df_filtered[df_filtered['h_a'] == 'h']
    elif match_type == 'Away':
        df_filtered = df_filtered[df_filtered['h_a'] == 'a']

    goals_per_week = df_filtered.groupby('matchweek')['scored'].sum().reset_index()

    fig = px.bar(
        goals_per_week,
        x='matchweek',
        y='scored',
        title=f"Matchweek-wise Distribution of Goals ({match_type.capitalize()})",
        labels={'matchweek': 'Matchweek', 'scored': 'Total Goals Scored'},
        color='scored',
        color_discrete_sequence=px.colors.qualitative.Set1 
    )

    fig.update_traces(text=goals_per_week['scored'], textposition='outside')

    fig.update_layout(
        xaxis_title='Matchweek',
        yaxis_title='Total Goals Scored',
        height=600,
        width=900,
        title_font_size=16,
        xaxis=dict(tickmode='linear'), 
        showlegend=False  
    )

    st.plotly_chart(fig)


# Number of points per matchweek
def plot_points_per_matchweek(year, league, match_type='Overall'):

    df_filtered = df_p[(df_p['year'] == year) & (df_p['league'] == league)]

    df_filtered['matchweek'] = df_filtered.groupby('team').cumcount() + 1

    if match_type == 'Home':
        df_filtered = df_filtered[df_filtered['h_a'] == 'h']
    elif match_type == 'Away':
        df_filtered = df_filtered[df_filtered['h_a'] == 'a']

    points_per_week = df_filtered.groupby('matchweek')['pts'].sum().reset_index()

    if points_per_week.empty:
        st.warning("No data available for the selected criteria.")
        return

    fig = px.bar(
        points_per_week,
        x='matchweek',
        y='pts',
        title=f"Total Points Scored per Matchweek ({match_type.capitalize()})",
        labels={'matchweek': 'Matchweek', 'pts': 'Total Points'},
        color='pts', 
        color_continuous_scale=px.colors.sequential.Viridis  
    )

    fig.update_traces(text=points_per_week['pts'], textposition='outside')

    fig.update_layout(
        xaxis_title='Matchweek',
        yaxis_title='Total Points',
        height=600,
        width=900,
        title_font_size=16,
        xaxis=dict(tickmode='linear'), 
        showlegend=False  
    )

    st.plotly_chart(fig)


# Comparison goals/xG
def plot_xg_per_matchweek(year, league, match_type='Overall'):

    df_filtered = df_p[(df_p['year'] == year) & (df_p['league'] == league)]
    
    df_filtered['matchweek'] = df_filtered.groupby('team').cumcount() + 1
    
    if match_type == 'Home':
        df_filtered = df_filtered[df_filtered['h_a'] == 'h']
    elif match_type == 'Away':
        df_filtered = df_filtered[df_filtered['h_a'] == 'a']

    goals_per_week = df_filtered.groupby('matchweek')['scored'].sum().reset_index()
    xG_per_week = df_filtered.groupby('matchweek')['xG'].sum().reset_index()
    merged_data = goals_per_week.merge(xG_per_week, on='matchweek', suffixes=('_goals', '_xG'))

    if merged_data.empty:
        st.warning("No data available for the selected criteria.")
        return

    goals_trace = go.Bar(
        x=merged_data['matchweek'],
        y=merged_data['scored'],
        name='Goals',
        marker=dict(color='blue')
    )
    
    xg_trace = go.Bar(
        x=merged_data['matchweek'],
        y=merged_data['xG'],  
        name='xG',
        marker=dict(color='orange')
    )

    fig = go.Figure(data=[goals_trace, xg_trace])
    fig.update_layout(
        title=f"Distribution of Goals and xG per Matchweek ({match_type.capitalize()})",
        xaxis_title="Matchweek",
        yaxis_title="Total Goals / xG",
        barmode='group'  
    )

    st.plotly_chart(fig)

# Comparison missed/xGA
def plot_missed_per_matchweek(year, league, match_type='Overall'):

    df_filtered = df_p[(df_p['year'] == year) & (df_p['league'] == league)]
    
    df_filtered['matchweek'] = df_filtered.groupby('team').cumcount() + 1
    
    if match_type == 'Home':
        df_filtered = df_filtered[df_filtered['h_a'] == 'h']
    elif match_type == 'Away':
        df_filtered = df_filtered[df_filtered['h_a'] == 'a']

    missed_per_week = df_filtered.groupby('matchweek')['missed'].sum().reset_index()
    xGA_per_week = df_filtered.groupby('matchweek')['xGA'].sum().reset_index()
    merged_data = missed_per_week.merge(xGA_per_week, on='matchweek', suffixes=('_missed', '_xGA'))

    if merged_data.empty:
        st.warning("No data available for the selected criteria.")
        return
    
    missed_trace = go.Bar(
        x=merged_data['matchweek'],
        y=merged_data['missed'],
        name='Missed',
        marker=dict(color='red')
    )
    
    xga_trace = go.Bar(
        x=merged_data['matchweek'],
        y=merged_data['xGA'],
        name='xGA',  
        marker=dict(color='purple')
    )

    fig = go.Figure(data=[missed_trace, xga_trace])
    fig.update_layout(
        title=f"Distribution of Missed Goals and xGA per Matchweek ({match_type.capitalize()})",
        xaxis_title="Matchweek",
        yaxis_title="Total Missed Goals / xGA",
        barmode='group' 
    )

    st.plotly_chart(fig)

# Comparison pts/xPTS
def plot_xpts_per_matchweek(year, league, match_type='Overall'):

    df_filtered = df_p[(df_p['year'] == year) & (df_p['league'] == league)]
    
    df_filtered['matchweek'] = df_filtered.groupby('team').cumcount() + 1
    
    if match_type == 'Home':
        df_filtered = df_filtered[df_filtered['h_a'] == 'h']
    elif match_type == 'Away':
        df_filtered = df_filtered[df_filtered['h_a'] == 'a']

    pts = df_filtered.groupby('matchweek')['pts'].sum().reset_index()
    xpts = df_filtered.groupby('matchweek')['xpts'].sum().reset_index()
    merged_data = pts.merge(xpts, on='matchweek', suffixes=('_pts', '_xpts'))

    if merged_data.empty:
        st.warning("No data available for the selected criteria.")
        return

    pts_trace = go.Bar(
        x=merged_data['matchweek'],
        y=merged_data['pts'],
        name='Points',
        marker=dict(color='blue')
    )
    
    xpts_trace = go.Bar(
        x=merged_data['matchweek'],
        y=merged_data['xpts'],
        name='Expected Points (xPTS)',
        marker=dict(color='orange')
    )

    fig = go.Figure(data=[pts_trace, xpts_trace])
    fig.update_layout(
        title=f"Distribution of Points and xPTS by Matchweek ({match_type.capitalize()})",
        xaxis_title="Matchweek",
        yaxis_title="Total Points / xPTS",
        barmode='group'  
    )
    st.plotly_chart(fig)


# Create a pie chart for win/draw/loss distribution
def plot_wdl_distribution(year, league):

    df_filtered = df_u[(df_u['Year'] == year) & (df_u['League'] == league)]
    
    if df_filtered.empty:
        st.warning(f"No data found for the year {year} and league {league}.")
        return
    
    wdl_counts = df_filtered.groupby('team').agg({
        'wins': 'sum',
        'draws': 'sum',
        'loses': 'sum'
    }).sum().reset_index()
    
    wdl_counts = wdl_counts.melt(var_name='Result', value_name='Count').set_index('Result')
    
    fig = px.pie(wdl_counts, values='Count', names=wdl_counts.index, title=f'Win/Draw/Loss Distribution for {league} in {year}')
    st.plotly_chart(fig)

###################################################################################################
    
###################################################################################################
    
###################################################################################################
    
# Home/Away Performance 

# Function to plot home and away points comparison 
def plot_home_away_points_barchart(league_name, year_selected):

    league_data = df_p[(df_p['league'] == league_name) & (df_p['year'] == year_selected)]

    grouped_data = league_data.groupby(['team', 'h_a'])['pts'].sum().unstack()

    grouped_data['total_pts'] = grouped_data['h'] + grouped_data['a']

    grouped_data = grouped_data.sort_values(by='total_pts', ascending=False)

    grouped_data = grouped_data[['h', 'a']].reset_index()
    grouped_data_melted = grouped_data.melt(id_vars='team', value_vars=['h', 'a'], 
                                            var_name='Home/Away', value_name='Points')

    fig = px.bar(grouped_data_melted, 
                 x='team', 
                 y='Points', 
                 color='Home/Away', 
                 title=f"Home and Away Points Comparison ({year_selected}) for Each Team of {league_name}",
                 labels={'team': 'Team', 'Points': 'Points'},
                 hover_data={'Points': True},
                 barmode='stack',  
                 text='Points' 
                 )

    fig.update_layout(xaxis_title="Team", 
                      yaxis_title="Points", 
                      legend_title="Home/Away",
                      xaxis_tickangle=-45, 
                      )

    st.plotly_chart(fig)

# Function to plot home and away wins comparison 
def plot_home_away_wins_barchart(league_name, year_selected):

    league_data = df_p[(df_p['league'] == league_name) & (df_p['year'] == year_selected)]

    grouped_data = league_data.groupby(['team', 'h_a'])['wins'].sum().unstack()

    grouped_data['total_wins'] = grouped_data['h'] + grouped_data['a']

    grouped_data = grouped_data.sort_values(by='total_wins', ascending=False)
    grouped_data = grouped_data[['h', 'a']].reset_index()
    grouped_data_melted = grouped_data.melt(id_vars='team', value_vars=['h', 'a'], 
                                            var_name='Home/Away', value_name='Wins')

    fig = px.bar(grouped_data_melted, 
                 x='team', 
                 y='Wins', 
                 color='Home/Away', 
                 title=f"Home and Away Wins Comparison ({year_selected}) for Each Team of {league_name}",
                 labels={'team': 'Team', 'wins': 'Wins'},
                 hover_data={'Wins': True},
                 barmode='stack',  
                 text='Wins' 
                 )

    fig.update_layout(xaxis_title="Team", 
                      yaxis_title="Wins", 
                      legend_title="Home/Away",
                      xaxis_tickangle=-45, 
                      )

    st.plotly_chart(fig)

# Function to plot home and away draws comparison 
def plot_home_away_draws_barchart(league_name, year_selected):

    league_data = df_p[(df_p['league'] == league_name) & (df_p['year'] == year_selected)]

    grouped_data = league_data.groupby(['team', 'h_a'])['draws'].sum().unstack()

    grouped_data['total_draws'] = grouped_data['h'] + grouped_data['a']

    grouped_data = grouped_data.sort_values(by='total_draws', ascending=False)

    grouped_data = grouped_data[['h', 'a']].reset_index()
    grouped_data_melted = grouped_data.melt(id_vars='team', value_vars=['h', 'a'], 
                                            var_name='Home/Away', value_name='Draws')

    fig = px.bar(grouped_data_melted, 
                 x='team', 
                 y='Draws', 
                 color='Home/Away', 
                 title=f"Home and Away Draws Comparison ({year_selected}) for Each Team of {league_name}",
                 labels={'team': 'Team', 'draws': 'Draws'},
                 hover_data={'Draws': True},
                 barmode='stack',  
                 text='Draws'  
                 )

    fig.update_layout(xaxis_title="Team", 
                      yaxis_title="Draws", 
                      legend_title="Home/Away",
                      xaxis_tickangle=-45,  
                      )

    st.plotly_chart(fig)

# Function to plot home and away losses comparison 
def plot_home_away_losses_barchart(league_name, year_selected):

    league_data = df_p[(df_p['league'] == league_name) & (df_p['year'] == year_selected)]

    grouped_data = league_data.groupby(['team', 'h_a'])['loses'].sum().unstack()

    grouped_data['total_loses'] = grouped_data['h'] + grouped_data['a']

    grouped_data = grouped_data.sort_values(by='total_loses', ascending=False)

    grouped_data = grouped_data[['h', 'a']].reset_index()
    grouped_data_melted = grouped_data.melt(id_vars='team', value_vars=['h', 'a'], 
                                            var_name='Home/Away', value_name='Losses')

    fig = px.bar(grouped_data_melted, 
                 x='team', 
                 y='Losses', 
                 color='Home/Away', 
                 title=f"Home and Away Losses Comparison ({year_selected}) for Each Team of {league_name}",
                 labels={'team': 'Team', 'losses': 'Losses'},
                 hover_data={'Losses': True},
                 barmode='stack', 
                 text='Losses'  
                 )

    fig.update_layout(xaxis_title="Team", 
                      yaxis_title="Losses", 
                      legend_title="Home/Away",
                      xaxis_tickangle=-45, 
                      )

    st.plotly_chart(fig)

# Function to plot home and away goals comparison 
def plot_home_away_goals_barchart(league_name, year_selected):

    league_data = df_p[(df_p['league'] == league_name) & (df_p['year'] == year_selected)]

    grouped_data = league_data.groupby(['team', 'h_a'])['scored'].sum().unstack()

    grouped_data['total_scored'] = grouped_data['h'] + grouped_data['a']

    grouped_data = grouped_data.sort_values(by='total_scored', ascending=False)

    grouped_data = grouped_data[['h', 'a']].reset_index()
    grouped_data_melted = grouped_data.melt(id_vars='team', value_vars=['h', 'a'], 
                                            var_name='Home/Away', value_name='scored')

    fig = px.bar(grouped_data_melted, 
                 x='team', 
                 y='scored', 
                 color='Home/Away', 
                 title=f"Home and Away Scored Comparison ({year_selected}) for Each Team of {league_name}",
                 labels={'team': 'Team', 'scored': 'Scored'},
                 hover_data={'scored': True},
                 barmode='stack',  
                 text='scored'  
                 )

    fig.update_layout(xaxis_title="Team", 
                      yaxis_title="Scored", 
                      legend_title="Home/Away",
                      xaxis_tickangle=-45, 
                      )

    st.plotly_chart(fig)

# Function to plot home and away misses comparison 
def plot_home_away_missed_barchart(league_name, year_selected):

    league_data = df_p[(df_p['league'] == league_name) & (df_p['year'] == year_selected)]

    grouped_data = league_data.groupby(['team', 'h_a'])['missed'].sum().unstack()

    grouped_data['total_missed'] = grouped_data['h'] + grouped_data['a']

    grouped_data = grouped_data.sort_values(by='total_missed', ascending=False)

    grouped_data = grouped_data[['h', 'a']].reset_index()
    grouped_data_melted = grouped_data.melt(id_vars='team', value_vars=['h', 'a'], 
                                            var_name='Home/Away', value_name='Missed')

    fig = px.bar(grouped_data_melted, 
                 x='team', 
                 y='Missed', 
                 color='Home/Away', 
                 title=f"Home and Away Missed Comparison ({year_selected}) for Each Team of {league_name}",
                 labels={'team': 'Team', 'missed': 'Missed'},
                 hover_data={'Missed': True},
                 barmode='stack',  
                 text='Missed'  
                 )

    fig.update_layout(xaxis_title="Team", 
                      yaxis_title="Missed", 
                      legend_title="Home/Away",
                      xaxis_tickangle=-45,  
                      )

    st.plotly_chart(fig)

###################################################################################################
    
###################################################################################################
    
###################################################################################################
    
# Pie Chart 

# Function to plot pie chart of goals scored by each team in a specific league and year
def plot_goals_scored_piecharts(league_name, year_selected):
    league_data = df_u[(df_u['League'] == league_name) & (df_u['Year'] == year_selected)]

    points_data = league_data[['team', 'scored']]

    fig = px.pie(points_data, 
                 values='scored', 
                 names='team', 
                 title=f"Goals Distribution by Team in {year_selected} - {league_name}",
                 color_discrete_sequence=px.colors.sequential.RdBu,  
                 )

    fig.update_traces(textinfo='percent+label', hoverinfo='label+value')

    st.plotly_chart(fig)

# Function to plot pie chart of points by each team in a specific league and year
def plot_goals_points_piecharts(league_name, year_selected):

    league_data = df_u[(df_u['League'] == league_name) & (df_u['Year'] == year_selected)]

    points_data = league_data[['team', 'pts']]

    fig = px.pie(points_data, 
                 values='pts', 
                 names='team', 
                 title=f"Points Distribution by Team in {year_selected} - {league_name}",
                 color_discrete_sequence=px.colors.sequential.RdBu, 
                 )

    fig.update_traces(textinfo='percent+label', hoverinfo='label+value')

    st.plotly_chart(fig)

# Function to plot pie chart of wins by each team in a specific league and year
def plot_goals_wins_piecharts(league_name, year_selected):

    league_data = df_u[(df_u['League'] == league_name) & (df_u['Year'] == year_selected)]

    points_data = league_data[['team', 'wins']]

    fig = px.pie(points_data, 
                 values='wins', 
                 names='team', 
                 title=f"Wins Distribution by Team in {year_selected} - {league_name}",
                 color_discrete_sequence=px.colors.sequential.RdBu,  
                 )

    fig.update_traces(textinfo='percent+label', hoverinfo='label+value')

    st.plotly_chart(fig)

# Function to plot pie chart of draws by each team in a specific league and year
def plot_goals_draws_piecharts(league_name, year_selected):

    league_data = df_u[(df_u['League'] == league_name) & (df_u['Year'] == year_selected)]

    points_data = league_data[['team', 'draws']]

    fig = px.pie(points_data, 
                 values='draws', 
                 names='team', 
                 title=f"Draws Distribution by Team in {year_selected} - {league_name}",
                 color_discrete_sequence=px.colors.sequential.RdBu,  
                 )

    fig.update_traces(textinfo='percent+label', hoverinfo='label+value')

    st.plotly_chart(fig)

# Function to plot pie chart of losses by each team in a specific league and year
def plot_goals_losses_piecharts(league_name, year_selected):

    league_data = df_u[(df_u['League'] == league_name) & (df_u['Year'] == year_selected)]

    points_data = league_data[['team', 'loses']]

    fig = px.pie(points_data, 
                 values='loses', 
                 names='team', 
                 title=f"Losses Distribution by Team in {year_selected} - {league_name}",
                 color_discrete_sequence=px.colors.sequential.RdBu,  
                 )

    fig.update_traces(textinfo='percent+label', hoverinfo='label+value')

    st.plotly_chart(fig)

# Function to plot pie chart of missed goals by each team in a specific league and year
def plot_goals_missed_piecharts(league_name, year_selected):

    league_data = df_u[(df_u['League'] == league_name) & (df_u['Year'] == year_selected)]

    points_data = league_data[['team', 'missed']]

    fig = px.pie(points_data, 
                 values='missed', 
                 names='team', 
                 title=f"Missed Goals Distribution by Team in {year_selected} - {league_name}",
                 color_discrete_sequence=px.colors.sequential.RdBu, 
                 )

    fig.update_traces(textinfo='percent+label', hoverinfo='label+value')

    st.plotly_chart(fig)

###################################################################################################
    
###################################################################################################
    
###################################################################################################
    
# PPDA/OPPDA 
    
deneme = df_p.groupby(["league","year"]).mean(numeric_only=True)
deneme = deneme.reset_index()
deneme['year'] = deneme['year'].astype(str)

deneme2=df_p.groupby(["league","team","year"]).mean(numeric_only=True)
deneme2=deneme2.reset_index()

# Comparison Evolution PPDA/year
def plot_ppda_coef(selected_league):
 
    filtered_data = deneme[deneme['league'] == selected_league]
    
    fig = px.bar(filtered_data, 
                 x='year', 
                 y='ppda_coef', 
                 color='year',
                 text='ppda_coef',
                 labels={"ppda_coef": "PPDA Coef Parameter Values", "league": "Leagues"},
                 title=f"PPDA Coef Value For {selected_league} Through Year",
                 color_discrete_sequence=px.colors.qualitative.Set3)
    
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(
        plot_bgcolor='#212121',
        paper_bgcolor='#212121',
        font=dict(color='white'),
        title=dict(text=f"PPDA Coeff Value For {selected_league} Through Year", font=dict(color='white', size=30)),
        xaxis_title='Leagues',
        yaxis_title='PPDA Coef Parameter Values',
        xaxis_tickfont=dict(color='white'),
        yaxis_tickfont=dict(color='white')
    )
    st.plotly_chart(fig)

def calculate_evolution_ratio(selected_league):

    league_data = deneme2[deneme2['league'] == selected_league]
    
    yearly_mean = league_data.groupby("year")["ppda_coef"].mean()

    evolution_ratio = yearly_mean.pct_change() * 100 
    evolution_ratio = evolution_ratio.dropna()

    st.subheader(f"Evolution Ratio of PPDA Coef for {selected_league}")
    st.dataframe(evolution_ratio)

def calculate_final_evolution_ratio(selected_league):

    league_data = deneme2[deneme2['league'] == selected_league]

    yearly_mean = league_data.groupby("year")["ppda_coef"].mean()
    value_2014 = yearly_mean.get(2014, np.nan)  # Value for 2014
    value_2019 = yearly_mean.get(2019, np.nan)  # Value for 2019

    if pd.notna(value_2014) and pd.notna(value_2019):
        final_evolution = ((value_2019 - value_2014) / value_2014) * 100  
    else:
        final_evolution = np.nan 

    st.write(f"The evolution ratio from 2014 to 2019 is: {final_evolution:.2f}%")
    


# Comparison Evolution Deep/year
def plot_deep_coef(selected_league):
 
    filtered_data = deneme[deneme['league'] == selected_league]
    
    fig = px.bar(filtered_data, 
                 x='year', 
                 y='deep',  
                 color='year',
                 text='deep',
                 labels={"deep": "Deep Coef Parameter Values", "league": "Leagues"},
                 title=f"Deep Coef Value For {selected_league} Through Year", 
                 color_discrete_sequence=px.colors.qualitative.Set3)
    
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(
        plot_bgcolor='#212121',
        paper_bgcolor='#212121',
        font=dict(color='white'),
        title=dict(text=f"Deep Coeff Value For {selected_league} Through Year", font=dict(color='white', size=30)),  
        xaxis_title='Leagues',
        yaxis_title='Deep Coef Parameter Values',  
        xaxis_tickfont=dict(color='white'),
        yaxis_tickfont=dict(color='white')
    )
    st.plotly_chart(fig)

# PPDA/OPPDA 2014 to 2019
def plot_ppda_oppda_comparison(selected_league):

    filtered_data = deneme2[(deneme2["league"] == selected_league) & ((deneme2['year'] == 2014) | (deneme2['year'] == 2019))]

    plt.figure(figsize=(25, 15), facecolor='black')
    plt.clf()  

    scatter_plot = plt.scatter(
        filtered_data["ppda_coef"],
        filtered_data["oppda_coef"],
        s=500,
        alpha=0.8,
        c=filtered_data["team"].astype('category').cat.codes,  
        cmap='rainbow',  
        edgecolor='black'
    )

    for i, row in filtered_data.iterrows():
        plt.text(row["ppda_coef"], row["oppda_coef"] + 0.4, f"{row['team']} {row['year']}", 
                 ha="center", color="black")

    for team in filtered_data["team"].unique():
        if (filtered_data[(filtered_data["team"] == team) & (filtered_data["year"] == 2014)].shape[0] != 0 and
            filtered_data[(filtered_data["team"] == team) & (filtered_data["year"] == 2019)].shape[0] != 0):
            x1 = filtered_data[(filtered_data["team"] == team) & (filtered_data["year"] == 2014)]["ppda_coef"].values
            y1 = filtered_data[(filtered_data["team"] == team) & (filtered_data["year"] == 2014)]["oppda_coef"].values
            x2 = filtered_data[(filtered_data["team"] == team) & (filtered_data["year"] == 2019)]["ppda_coef"].values
            y2 = filtered_data[(filtered_data["team"] == team) & (filtered_data["year"] == 2019)]["oppda_coef"].values
            
            norm = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** (1 / 2)
            plt.plot([x1, x2], [y1, y2], color='#00CED1', alpha=0.3, linewidth=4) 
            plt.quiver(
                x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2,
                (x2 - x1) / norm, (y2 - y1) / norm,
                angles="xy", scale=20, headwidth=2, headlength=4.5,
                linewidth=0.5, color='#00CED1', alpha=0.5, pivot="mid"
            )

    plt.grid(color='black')
    plt.xlabel("Average ppda_coef", fontsize=20, color="white")
    plt.ylabel("Average oppda_coef", fontsize=20, color="white")
    plt.xticks(rotation=90)
    plt.title(f"Comparison PPDA/OPPDA for {selected_league} from 2014 to 2019", fontsize=30, color="white")
    plt.gca().invert_xaxis()
    plt.tick_params(axis='x', colors='white')
    plt.tick_params(axis='y', colors='white')

    st.pyplot(plt, clear_figure=True)
    plt.clf()

# Function to plot correlation heatmap for a specific league and year
def plot_correlation_heatmap(selected_league):

    df = df_u[df_u['League'] == selected_league]
    df = df[['scored', 'missed', 'ppda_coef', 'oppda_coef', 'deep', 'deep_allowed']]

    if df.empty:
        st.warning(f"No data found for League: {selected_league}")
        return

    corrmat = df.corr()

    fig = px.imshow(corrmat,
                     text_auto=True, 
                     color_continuous_scale='RdYlGn',  
                     labels=dict(color='Correlation'),
                     )

    fig.update_xaxes(side="top")  
    fig.update_layout(
        title=f"Correlation Heatmap for {selected_league}",
        plot_bgcolor='#212121',
        paper_bgcolor='#212121',
        font=dict(color='white')
    )

    st.plotly_chart(fig)

# Function to display comparison
def display_comparison(selected_league):

    subsets = ['scored', 'xG', 'ppda_coef', 'deep', 'pts']
    
    league_data = df_p[df_p['league'] == selected_league]
    league_data = league_data[subsets]

    other_data = df_p[df_p['league'] != selected_league]
    other_data = other_data[subsets]

    tab_names = ['Scored', 'xG', 'PPDA Coefficient', 'Deep', 'Points']
    tabs = st.tabs(tab_names)

    for idx, metric in enumerate(subsets):
        with tabs[idx]:
            avg_league = round(league_data[metric].mean(), 3) * 2
            avg_other = round(other_data[metric].mean(), 3) * 2

            st.write(f"Average **{metric}** in {selected_league} = {avg_league}")
            st.write(f"Average **{metric}** in Other Leagues = {avg_other}")
            st.write("\n")  

###################################################################################################
            
###################################################################################################
            
###################################################################################################
            
# WWW and LL

def analyze_top_4_performance(league):
    
    top_4_count = df_u[(df_u['League'] == league) & (df_u['position'] <= 4)]

    if top_4_count.empty:
        st.write(f"No data found for league: {league}")
        return

    top_4_team_count = top_4_count.groupby('team').agg(
        top_4_count=('position', 'size'),
        years_in_top_4=('Year', list)
    ).reset_index().sort_values(by='top_4_count', ascending=False)
    
    top_4_team_count.rename(columns={
        'team': 'Team',
        'top_4_count': 'Top 4 Count',
        'years_in_top_4': 'Years in Top 4'
    }, inplace=True)

    top_4_team_count.set_index('Team',inplace=True)

    st.write("### Top 4 Performance Count")
    st.dataframe(top_4_team_count)

    fig = px.bar(top_4_count, 
                 x='team', 
                 y='pts', 
                 color='Year',  
                 title=f'Team Points in {league} (Top 4)', 
                 labels={'pts': 'Points', 'team': 'Team', 'Year': 'Year'},
                 text='pts',
                 color_discrete_sequence=px.colors.qualitative.Plotly)  
    
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(xaxis_tickangle=-45, 
                      width=1200, 
                      height=600)
    
    st.plotly_chart(fig)

def plot_points_comparison(league, statname):

    if statname == 'Winners':
        teams = df_u[df_u['position'] == 1]
        title_suffix = "Winner"
    elif statname == 'Losers':
        teams = df_u[df_u['position'] == df_u['position'].max()] 
        title_suffix = "Loser"
    else:
        st.error("Invalid statname. Please use 'winners' or 'losers'.")
        return

    years = df_u['Year'].drop_duplicates().tolist()

    pts = go.Bar(x=years, y=teams['pts'][teams['League'] == league], name='PTS')
    xpts = go.Bar(x=years, y=teams['xpts'][teams['League'] == league], name='Expected PTS')

    data = [pts, xpts]

    layout = go.Layout(
        barmode='group',
        title=f"Comparing Actual and Expected Points for {title_suffix} Team in {league}",
        xaxis={'title': 'Year'},
        yaxis={'title': "Points"},
    )

    fig = go.Figure(data=data, layout=layout)
    
    st.plotly_chart(fig)

def plot_goals_comparison(league, statname):

    if statname == 'Winners':
        teams = df_u[df_u['position'] == 1]
        title_suffix = "Winner"
    elif statname == 'Losers':
        teams = df_u[df_u['position'] == df_u['position'].max()]  
        title_suffix = "Loser"
    else:
        st.error("Invalid statname. Please use 'winners' or 'losers'.")
        return

    years = df_u['Year'].drop_duplicates().tolist()

    goals = go.Bar(x=years, y=teams['scored'][teams['League'] == league], name='Goals')
    xg = go.Bar(x=years, y=teams['xG'][teams['League'] == league], name='Expected Goals')

    data = [goals, xg]

    layout = go.Layout(
        barmode='group',
        title=f"Comparing Actual and Expected Goals for {title_suffix} Team in {league}",
        xaxis={'title': 'Year'},
        yaxis={'title': "Goals"},
    )

    fig = go.Figure(data=data, layout=layout)
    
    st.plotly_chart(fig)


def plot_missed_comparison(league, statname):

    if statname == 'Winners':
        teams = df_u[df_u['position'] == 1]
        title_suffix = "Winner"
    elif statname == 'Losers':
        teams = df_u[df_u['position'] == df_u['position'].max()]  
        title_suffix = "Loser"
    else:
        st.error("Invalid statname. Please use 'winners' or 'losers'.")
        return

    years = df_u['Year'].drop_duplicates().tolist()

    missed = go.Bar(x=years, y=teams['missed'][teams['League'] == league], name='Missed')
    xga = go.Bar(x=years, y=teams['xGA'][teams['League'] == league], name='Expected Missed')

    data = [missed, xga]

    layout = go.Layout(
        barmode='group',
        title=f"Comparing Actual and Expected Missed for {title_suffix} Team in {league}",
        xaxis={'title': 'Year'},
        yaxis={'title': "Goals"},
    )

    fig = go.Figure(data=data, layout=layout)
    
    st.plotly_chart(fig)

def get_records_antirecords(league):

    records = []  
    
    league_df = df_u[df_u['League'] == league]
    
    for col in league_df.describe().columns:
        if col not in ['index', 'Year', 'position']:
            team_min = league_df['team'].loc[league_df[col] == league_df.describe().loc['min', col]].values[0]
            year_min = league_df['Year'].loc[league_df[col] == league_df.describe().loc['min', col]].values[0]
            team_max = league_df['team'].loc[league_df[col] == league_df.describe().loc['max', col]].values[0]
            year_max = league_df['Year'].loc[league_df[col] == league_df.describe().loc['max', col]].values[0]
            val_min = league_df.describe().loc['min', col]
            val_max = league_df.describe().loc['max', col]
            
            records.append({
                'Statistic': col.upper(),
                'Lowest Team': team_min,
                'Lowest Year': year_min,
                'Lowest Value': val_min,
                'Highest Team': team_max,
                'Highest Year': year_max,
                'Highest Value': val_max
            })
    
    records_df = pd.DataFrame(records)
    records_df.set_index('Statistic',inplace=True)
    st.dataframe(records_df)

def plot_xg_difference(league):

    league_df = df_u[df_u['League'] == league]

    years = league_df['Year'].drop_duplicates().tolist()
    data = []  
    for year in years:
        trace = go.Scatter(
            x=league_df['position'][league_df['Year'] == year],
            y=league_df['xG_diff'][league_df['Year'] == year],
            name=str(year),
            mode='lines+markers'
        )
        data.append(trace)

    layout = go.Layout(
        title=f"Comparing xG Difference Between Positions in {league}",
        xaxis={'title': 'Position'},
        yaxis={'title': "xG Difference"},
    )

    fig = go.Figure(data=data, layout=layout)

    st.plotly_chart(fig)
    st.write("""
        **xG_diff = xG - G**
    - xG_diff > 0 = have scored less than expectation 
    - xG_diff < 0 = have scored more than expecation
    """)

def plot_xga_difference(league):

    league_df = df_u[df_u['League'] == league]

    years = league_df['Year'].drop_duplicates().tolist()
    data = []  

    for year in years:
        trace = go.Scatter(
            x=league_df['position'][league_df['Year'] == year],
            y=league_df['xGA_diff'][league_df['Year'] == year],
            name=str(year),
            mode='lines+markers'
        )
        data.append(trace)

    layout = go.Layout(
        title=f"Comparing xGA Difference Gap Between Positions in {league}",
        xaxis={'title': 'Position'},
        yaxis={'title': "xGA Difference"},
    )

    fig = go.Figure(data=data, layout=layout)

    st.plotly_chart(fig)
    st.write("""
        **xGA_diff = xGA - GA**
    - xGA_diff > 0 = have conceded less than expected 
    - xGA_diff < 0 = have conceded more than expected
    """)

def plot_xpts_difference(league):

    league_df = df_u[df_u['League'] == league]

    years = league_df['Year'].drop_duplicates().tolist()
    data = []  

    for year in years:
        trace = go.Scatter(
            x=league_df['position'][league_df['Year'] == year],
            y=league_df['xpts_diff'][league_df['Year'] == year],
            name=str(year),
            mode='lines+markers'
        )
        data.append(trace)

    layout = go.Layout(
        title=f"Comparing xPTS Difference Gap Between Positions in {league}",
        xaxis={'title': 'Position'},
        yaxis={'title': "xPTS Difference"},
    )

    fig = go.Figure(data=data, layout=layout)

    st.plotly_chart(fig)
    st.write("""
        **xPTS_diff = xPTS - PTS**  
    - xPTS_diff > 0 = have won less than expected 
    - xPTS_diff < 0 = have won more than expected
    """)

def detect_outliers(statname):

    def iqr_outlier_detection(column):
        iqr = (df_u.describe().loc['75%', column] - df_u.describe().loc['25%', column]) * 1.5
        upper_bound = df_u.describe().loc['75%', column] + iqr
        lower_bound = df_u.describe().loc['25%', column] - iqr
        return upper_bound, lower_bound

    upper_xG, lower_xG = iqr_outlier_detection('xG_diff')
    outliers_xG = df_u[(df_u['xG_diff'] > upper_xG) | (df_u['xG_diff'] < lower_xG)]

    upper_xGA, lower_xGA = iqr_outlier_detection('xGA_diff')
    outliers_xGA = df_u[(df_u['xGA_diff'] > upper_xGA) | (df_u['xGA_diff'] < lower_xGA)]

    upper_xpts, lower_xpts = iqr_outlier_detection('xpts_diff')
    outliers_xpts = df_u[(df_u['xpts_diff'] > upper_xpts) | (df_u['xpts_diff'] < lower_xpts)]

    outliers_full = pd.concat([outliers_xG, outliers_xGA, outliers_xpts]).drop_duplicates()

    max_position = df_u['position'].max()
    df_u['position_reverse'] = max_position - df_u['position'] + 1
    outliers_full['position_reverse'] = max_position - outliers_full['position'] + 1

    if statname == "Overperformance":
        total_count = df_u[df_u['position'] <= 4].shape[0]
        outlier_count = outliers_full[outliers_full['position'] <= 4].shape[0]
        top_bottom = outliers_full[outliers_full['position'] <= 4].sort_values(by='League')
    elif statname == "Underperformance":
        total_count = df_u[df_u['position_reverse'] <= 3].shape[0]
        outlier_count = outliers_full[outliers_full['position_reverse'] <= 3].shape[0]
        top_bottom = outliers_full[outliers_full['position_reverse'] <= 3].sort_values(by='League')

    outlier_prob = (outlier_count / total_count) * 100 if total_count > 0 else 0

    if statname == "Overperformance":
        st.write(f"Probability of outlier in top of the final table: {outlier_prob}%")
    elif statname == 'Underperformance':
        st.write(f"Probability of outlier in bottom of the final table: {outlier_prob}%")

    top_bottom.set_index('League',inplace=True)
    st.dataframe(top_bottom)

    outliers_per_league = outliers_full.groupby('League').size().reset_index(name='Outlier Count')

    league_colors = {
        'EPL': 'red',
        'La_liga': 'blue',
        'Serie_A': 'green',
        'Bundesliga': 'yellow',
        'Ligue_1': 'purple',
        'RFPL': 'orange'
    }

    bar_colors = [league_colors.get(league, 'gray') for league in outliers_per_league['League']]

    fig = go.Figure(go.Bar(
        x=outliers_per_league['League'],
        y=outliers_per_league['Outlier Count'],
        marker_color=bar_colors  
    ))

    fig.update_layout(
        title="Number of Outliers per League",
        xaxis_title="League",
        yaxis_title="Outlier Count"
    )

    st.plotly_chart(fig)

    st.write("""
        In this analysis, team performances (xG, xGA, xPTS) were assessed using **z-score** and **Interquartile Range (IQR)** to detect outliers.

        **Z-Score**:
             
        The z-score shows how far a value is from the average, measured in standard deviations. Teams with a z-score above 3 or below -3 are flagged as significant outliers, indicating extreme over- or underperformance compared to the league average.

        **IQR**:
        The IQR measures the spread of the middle 50 percent of data. Outliers are values that fall 1.5 times beyond the IQR, helping to spot unusual performances.

        These methods help identify teams with notable over- or underperformance in the league.
    """)


###################################################################################################
            
###################################################################################################
            
###################################################################################################

# Playing Styles 

league_names_mapping_clu = { 
    'EPL': 'Premier League',
    'La_liga': 'La Liga',
    'Bundesliga': 'Bundesliga',}

df_understat = pd.read_csv('analyzing_football_european_leagues/understat.com.csv')
#df_understat = pd.read_csv('understat.com.csv')
df_2= pd.read_csv('analyzing_football_european_leagues/Complete_Dataset_2.csv')
#df_2= pd.read_csv('Complete_Dataset_2.csv')
    
# Preprocessing
df_2["Start Season"] = df_2["Season"].apply(lambda x: int(x[:4]))
df_2["End Season"] = df_2["Season"].apply(lambda x: int(x[5:]))
df_2["League"].replace('La Liga', 'La_liga', inplace=True)
df_2["League"].replace('Premier League', 'EPL', inplace=True)

# Rename columns for understat dataframe
df_understat.rename(columns={
    "Unnamed: 0": "league",
    "Unnamed: 1": "season",
    "position": "Rank"
}, inplace=True)
            
def plot_team_clusters_by_year(league, year):
    league_data = df_understat[(df_understat["league"] == league) & (df_understat["season"] == year)][["Rank", "team", "npxGD"]].sort_values(["Rank"]).reset_index(drop=True)

    teams_data = df_2[(df_2["Start Season"] == year) & (df_2["League"] == league)][['Rank', 'League', 'Team', 'Possession', 'ShortPassesPerGame', 'LongBallsPerGame']].sort_values(["Rank"]).reset_index(drop=True)

    teams_data["SPPLBP"] = teams_data["ShortPassesPerGame"] / teams_data["LongBallsPerGame"]
    teams_data.drop(["LongBallsPerGame", "ShortPassesPerGame"], axis=1, inplace=True)
    teams_data["Rank"] = teams_data["Rank"].astype(int)
    teams_data["npxGD"] = league_data["npxGD"]

    df_prepared = teams_data

    scaler = MinMaxScaler()
    df_prepared[["Possession", "SPPLBP", "npxGD"]] = scaler.fit_transform(df_prepared[["Possession", "SPPLBP", "npxGD"]])

    # K-Means clustering
    n = 6
    kmeans_v2 = KMeans(n_clusters=n, max_iter=3000, random_state=42)
    kmeans_v2.fit(df_prepared[["Possession", "SPPLBP", "npxGD"]])
    df_prepared["cluster"] = kmeans_v2.labels_ + 1

    title_kmeans_v2 = f'Clustering {league} Teams for the Year {year} (K-Means, n = {n})'
    kmeans_v2_fig = px.scatter_3d(
        df_prepared,
        x='Possession',
        y='SPPLBP',
        z='npxGD',
        hover_name='Team',
        color='cluster',
        title=title_kmeans_v2
    )

    st.plotly_chart(kmeans_v2_fig)

    cluster_table = df_prepared[['Rank','Team','cluster']]
    cluster_table.set_index('Rank',inplace=True)

    st.dataframe(cluster_table)

    

# Create a dictionary with cluster descriptions
cluster_descriptions = {
    1: """
    **1. Dominant Powerhouses**

    **Playing Style:** High-intensity possession, relentless attacking, and complete control.

    - **Tactical Approach:** Teams in this category focus on total dominance, often using high pressing to regain possession quickly and maintain overwhelming control. They typically have elite players across all positions, giving them the ability to break down even the most resilient defenses. These teams are often aggressive in pushing forward, employing advanced tactics like inverted full-backs, overlapping center-backs, and attacking midfielders who operate between the lines to create numerical superiority.
    - **Strengths:** Clinical finishing, high possession, and dominance in both physical and technical aspects of the game.
    - **Weaknesses:** Vulnerability to counter-attacks if the pressing or positioning fails.
    """,
    2: """
    **2. Steady Mid-table Contenders**

    **Playing Style:** Organized, pragmatic, and balanced in attack and defense.

    - **Tactical Approach:** These teams typically focus on solid defensive organization while being opportunistic in attack. They don't take unnecessary risks, playing conservatively when needed but showing creativity when given space. Often these teams rely on counter-attacking football or structured build-up play to capitalize on chances when they arise. Their game plan is to maintain mid-table security and occasionally push for European spots.
    - **Strengths:** Discipline, tactical awareness, and adaptability to different match scenarios.
    - **Weaknesses:** Inconsistency when facing elite opposition, sometimes lacking the firepower or depth to consistently challenge top teams.
    """,
    3: """
    **3. Mid-table Battlers**

    **Playing Style:** Gritty, physical, and focused on grinding out results.

    - **Tactical Approach:** These teams often find themselves in physical battles and scrappy matches. They prioritize defensive solidity and rely on resilience and team spirit to secure points. Mid-table battlers play with a defensive-first mindset, frequently resorting to long balls, set-pieces, or counter-attacks to score. Their primary focus is avoiding relegation, making them difficult to break down and highly tenacious in defense.
    - **Strengths:** Defensive commitment, strong work ethic, and ability to frustrate more talented opponents.
    - **Weaknesses:** Limited attacking options and technical flair, often relying too heavily on defensive tactics.
    """,
    4: """
    **4. Title Contenders**

    **Playing Style:** Tactical versatility with elite attacking and defensive capabilities.

    - **Tactical Approach:** These teams have the quality to challenge for league titles and major trophies. They combine defensive solidity with a dynamic attacking approach, with star players across different positions who can adapt to different game plans. Title contenders often dominate smaller teams and are capable of managing big games against fellow top sides with strong defensive discipline or tactical flexibility. They also boast depth in their squad, allowing for a high level of consistency across multiple competitions.
    - **Strengths:** Balanced squads, tactical flexibility, and the ability to control games through possession or rapid transitions.
    - **Weaknesses:** High expectations sometimes lead to pressure in key games, and they can be vulnerable when faced with injuries to star players.
    """,
    5: """
    **5. Resilient Defenders**

    **Playing Style:** Deep defensive structure with a focus on soaking pressure and counter-attacking.

    - **Tactical Approach:** These teams build their identity around defensive robustness, often adopting a low block or a compact shape to minimize the space for opponents to attack. The strategy is to absorb pressure and look for quick counters or set-piece opportunities. Resilient defenders are happy to let the opponent dominate possession as long as they keep their defensive shape intact. Tactical fouling, strategic marking, and well-drilled defensive units are key components of their style.
    - **Strengths:** Strong defensive cohesion, great at frustrating attacking teams, and dangerous on counters or set-pieces.
    - **Weaknesses:** Limited creativity in attack, reliance on few chances to score, and risk of being pinned back in their half for long periods.
    """,
    6: """
    **6. Dynamic Midfield Enforcers**

    **Playing Style:** Midfield-dominant, focusing on dictating the pace of the game and pressing.

    - **Tactical Approach:** These teams rely heavily on their midfield to control both offensive and defensive phases of play. Their midfield enforcers are tasked with breaking up opposition attacks and launching their own through incisive passing or progressive runs. Teams with dynamic midfielders often press high up the pitch, forcing turnovers and looking to transition quickly into attacks. This style is key to maintaining possession, disrupting opponents, and asserting control over the game's tempo.
    - **Strengths:** High pressing, control of the game's tempo, and ability to recover the ball quickly.
    - **Weaknesses:** Can be exposed by teams with fast transitions if the midfielders are bypassed, leading to defensive vulnerability.
    """
}


###################################################################################################

###################################################################################################

###################################################################################################

###################################################################################################

###################################################################################################

###################################################################################################

# Web app code
    
st.set_page_config(page_icon="analyzing_football_european_leagues/img_dv/uefa.jpg", page_title="Football Visualizations", layout="wide")

st.write("""
         # Analyzing European Leagues through Advanced Statistical Metrics ⚽
         ## A visual deep dive into the last *5 years* of the **European Leagues** 
         """)
st.write('---')

# Sidebar
st.sidebar.image("analyzing_football_european_leagues/img_dv/uefa.jpg")
st.sidebar.markdown('---')

st.sidebar.header('Data-Driven Insights into European Leagues')
st.sidebar.markdown(
    """
- **Comprehensive Analysis** 
- **Data Visualization** 
- **Insights and Trends** 
"""
)
st.sidebar.markdown('---')

st.sidebar.markdown(
    """
All Rights Reserved - AntoTim
"""
)
st.sidebar.markdown('---')

# Sidebar buttons
cols = st.sidebar.columns(2)

cols[0].link_button('Linkedin', 'https://www.linkedin.com/in/anto-tim')

if cols[1].button('About Me'):
    st.session_state.show_about_me = True
else:
    st.session_state.show_about_me = False

st.sidebar.markdown("---")

if 'show_about_me' not in st.session_state:
    st.session_state.show_about_me = False

if st.session_state.show_about_me:
    st.write("""
        ## Anto TIM
        ### Student at EFREI Paris
        ### Engineering Program: Big Data and Machine Learning      

        ---

        **SKILLS**

        **TECHNICAL KNOWLEDGE**  
        - Procedural Programming (Python)  
        - Object-Oriented Programming (Java)  
        - Databases (SQL)  
        - Web Development (HTML, PHP, CSS, JavaScript)  
        - Statistics and Probability  
        - Real Analysis, Linear Algebra  
        - Economic Theories  
        - Numerical Methods  

        ---

        **PROJECTS**

        **WEBSITE CREATION**  
        Development and management of a 2-month school project  
        - Developed a website for a fictional real estate agency  
        - Created a project specification and timeline  
        - Delivered a PowerPoint presentation to the client  
        - Programming languages used: HTML, CSS, PHP, and JavaScript  

        **TRADING BOT CREATION**  
        Programmer for a personal project  
        - Developed a cryptocurrency trading bot on Binance  
        - Created a chatbot for tracking currencies and their fluctuations  
        - Utilized libraries such as NumPy, Pandas, and Matplotlib  

        **SOLUTION FACTORY - EXPLAIN**  
        Team leader for a school project lasting one month  
        - Developed an explainable text classification method  
        - Used deep learning models  
        - Planned and organized various tasks within the team  

        ---

        **EDUCATION**

        **2024**  
        AGH UNIVERSITY OF SCIENCE AND TECHNOLOGY, Krakow (30-059, Lesser Poland)  
        - Semester abroad: General Computer Science  

        **Since 2023**  
        EFREI PARIS, Villejuif (94800)  
        - Engineering Program: General Computer Science  

        **2023**  
        CALIFORNIA STATE UNIVERSITY LONG BEACH  
        - Semester abroad: Mathematics, Finance, and Insurance  
        Long Beach (90802, CA)  

        **2019-2023**  
        UNIVERSITY OF PARIS 1 PANTHEON SORBONNE, Paris (75013)  
        - Master 1 MAEF: Financial Mathematics  
        - Bachelor in MIASHS: Statistics, Economics, Programming  

        """)

    st.markdown('---')
    
    if st.button('Go Back to Home'):
        st.session_state.show_about_me = False  

else:
    # Tabs
    tabs = ['League Table', 'Season Progress Chart', 'Matchweek Analysis', 'Home/Away Performance', 'Pie Chart', 'PPDA/OPPDA', 'Why Winners Win and Losers Lose', 'Playing Style']
    tab_list = st.tabs(tabs)


    # League Table
    with tab_list[0]:
        st.header("League Table")
        league_display_names = list(league_names_mapping.values())
        league_choice = st.selectbox('Choose the league', league_display_names,  key='league_table_choice')
        league_selected = [key for key, value in league_names_mapping.items() if value == league_choice][0]
        year = st.selectbox('Choose the year', df_u['Year'].unique(), key='league_table_year')
        plot_league_table_by_year_and_league(year, league_selected)

    # Season Progress Chart 
    with tab_list[1]:
        st.header("Season Progress Chart")
        league_display_names = list(league_names_mapping.values())
        league_choice = st.selectbox('Choose the league', league_display_names, key='season_progress_choice')
        league_selected = [key for key, value in league_names_mapping.items() if value == league_choice][0]
        year = st.selectbox('Choose the year', df_u['Year'].unique(), key='season_progress_year')
        plot_final_standings_and_trend(year, league_selected)

    # Matchweek Analysis
    with tab_list[2]:
        st.header("Matchweek Analysis")
        statname = st.selectbox('Statistic', ('Goals', 'Points', 'Expected Goals', 'Expected Goals Against', 'Expected Points'), key='matchweek_analysis_statname')
        league_display_names = list(league_names_mapping.values())
        league_choice = st.selectbox('Choose the league', league_display_names, key='matchweek_analysis_choice')
        league_selected = [key for key, value in league_names_mapping.items() if value == league_choice][0]
        year = st.selectbox('Choose the year', df_u['Year'].unique(), key='matchweek_analysis_year')
        match_type = st.radio('Mode', ('Overall', 'Home', 'Away'), horizontal=True, key='matchweek_analysis_mode')


        if statname == 'Goals':
            plot_goals_per_matchweek(year, league_selected, match_type)
        elif statname == 'Points':
            plot_points_per_matchweek(year, league_selected, match_type)
        elif statname == 'Expected Goals':
            plot_xg_per_matchweek(year, league_selected, match_type)
        elif statname == 'Expected Goals Against':
            plot_missed_per_matchweek(year, league_selected, match_type)
        elif statname == 'Expected Points':
            plot_xpts_per_matchweek(year, league_selected, match_type)

    # Home/Away Performance
    with tab_list[3]:
        st.header("Home/Away Performance")
        statname = st.selectbox('H/A Statistic', ('Points', 'Wins', 'Draws', 'Losses', 'Goals Scored', 'Goals Conceded'), key='home_away_statname')
        league_display_names = list(league_names_mapping.values())
        league_choice = st.selectbox('Choose the league', league_display_names, key='home_away_choice')
        league_selected = [key for key, value in league_names_mapping.items() if value == league_choice][0]
        year = st.selectbox('Choose the year', df_u['Year'].unique(), key='home_away_year')

        # Option to compare
        compare = st.selectbox('Compare with another league/season?', ('No Comparison', 'Yes'), key='compare_choice')
        
        if compare == 'Yes':
            compare_league_choice = st.selectbox('Choose the comparison league', league_display_names, key='compare_league_choice')
            compare_league_selected = [key for key, value in league_names_mapping.items() if value == compare_league_choice][0]

            if compare_league_selected == league_selected:
                available_years = [y for y in df_u['Year'].unique() if y != year]
            else:
                available_years = df_u['Year'].unique()
            
            compare_year = st.selectbox('Choose the comparison year', available_years, key='compare_year')

        if statname == 'Points':
            plot_home_away_points_barchart(league_selected, year)
            if compare == 'Yes':
                plot_home_away_points_barchart(compare_league_selected, compare_year)
        
        elif statname == 'Wins':
            plot_home_away_wins_barchart(league_selected, year)
            if compare == 'Yes':
                plot_home_away_wins_barchart(compare_league_selected, compare_year)

        elif statname == 'Draws':
            plot_home_away_draws_barchart(league_selected, year)
            if compare == 'Yes':
                plot_home_away_draws_barchart(compare_league_selected, compare_year)

        elif statname == 'Losses':
            plot_home_away_losses_barchart(league_selected, year)
            if compare == 'Yes':
                plot_home_away_losses_barchart(compare_league_selected, compare_year)

        elif statname == 'Goals Scored':
            plot_home_away_goals_barchart(league_selected, year)
            if compare == 'Yes':
                plot_home_away_goals_barchart(compare_league_selected, compare_year)

        elif statname == 'Goals Conceded':
            plot_home_away_missed_barchart(league_selected, year)
            if compare == 'Yes':
                plot_home_away_missed_barchart(compare_league_selected, compare_year)

    # Pie Chart 
    with tab_list[4]:
        st.header("Pie Chart")
        statname = st.selectbox('Statistical Distribution', ('Points', 'Wins', 'Draws', 'Losses', 'Goals Scored', 'Goals Conceded'), key='pie_chart_statname')
        league_display_names = list(league_names_mapping.values())
        league_choice = st.selectbox('Choose the league', league_display_names, key='pie_chart_choice')
        league_selected = [key for key, value in league_names_mapping.items() if value == league_choice][0]
        year = st.selectbox('Choose the year', df_u['Year'].unique(), key='pie_chart_year')

        if statname == 'Points':
            plot_goals_points_piecharts(league_selected, year)
        elif statname == 'Wins':
            plot_goals_wins_piecharts(league_selected, year)
        elif statname == 'Draws':
            plot_goals_draws_piecharts(league_selected, year)
        elif statname == 'Losses':
            plot_goals_losses_piecharts(league_selected, year)
        elif statname == 'Goals Scored':
            plot_goals_scored_piecharts(league_selected, year)
        elif statname == 'Goals Conceded':
            plot_goals_missed_piecharts(league_selected, year)

    # PPDA/OPPDA
    with tab_list[5]:
        st.header("PPDA/OPPDA and Deep")
        statname = st.selectbox('Statistic', ('PPDA Coefficient', 'Deep Coefficient', 'PPDA/OPPDA Comparison', 'Heatmap Correlation'), key='ppda_oppda_statname')
        league_display_names = list(league_names_mapping.values())
        league_choice = st.selectbox('Choose the league', league_display_names, key='ppda_oppda_choice')
        league_selected = [key for key, value in league_names_mapping.items() if value == league_choice][0]

        # Option to compare
        compare = st.selectbox('Compare with another league/season?', ('No Comparison', 'Yes'), key='compare_choice_ppda')

        if compare == 'Yes':
            available_compare_leagues = [
            league for league in league_display_names if league != league_choice
            ]

            # Choose the comparison league
            compare_league_choice = st.selectbox(
            'Choose the comparison league',
            available_compare_leagues,  # Only show different leagues
            key='compare_league_choice_ppda'
            )

            compare_league_selected = [key for key, value in league_names_mapping.items() if value == compare_league_choice][0]


        if statname == 'PPDA Coefficient':
            st.write("""
                PPDA (Passes per Defensive Action) is a metric used to measure a team's pressing intensity, particularly in high-pressing tactics. 
                    
                It calculates how many passes a team allows their opponent to make before attempting a defensive action (such as a tackle, interception, or pressing the ball).
                    """)
            plot_ppda_coef(league_selected)
            if compare == 'Yes':
                plot_ppda_coef(compare_league_selected)
            st.write("""
                    Lower PPDA: Indicates that a team is applying high pressure on the opponent, forcing them to make fewer passes before the defensive team intervenes.
                    
                    Higher PPDA: Suggests that a team is allowing the opponent to pass more freely before engaging in defensive actions.
                    """)
            calculate_evolution_ratio(league_selected)
            calculate_final_evolution_ratio(league_selected)
        elif statname == 'Deep Coefficient':
            st.write("""
                "Deep" (or "Deep Value") refers to a metric that measures how often a team reaches dangerous areas of the pitch, typically within close proximity to the opponent's goal. 
                    
                It focuses on how often a team penetrates these critical areas, which are key to creating high-quality scoring opportunities.
                    
                Key Components Explained:
                    - Key Passes: Passes that directly lead to a shot on goal.
                    - Successful Dribbles: The number of times a player successfully takes on an opponent with a dribble.
                    - Defensive Actions in the Final Third: Defensive contributions (like tackles, interceptions, and clearances) made in the opponent's final third of the pitch.
                    - Goals: Total goals scored by a player.
                    - Assists: Total assists provided by a player.
                    - Turnovers: Instances where a player loses possession of the ball, which can negatively impact the team's performance.
                    """)
            plot_deep_coef(league_selected)
            if compare == 'Yes':
                plot_deep_coef(compare_league_selected)
            st.write("""
                    A higher Deep Value suggests a player is making more impactful contributions both offensively and defensively.
                    
                    A lower Deep Value might indicate that the player is less involved in critical actions that can influence the game positively.
                    """)
        elif statname == 'PPDA/OPPDA Comparison':
            st.write("""
                    OPPDA (Opponent Passes per Defensive Action) is the inverse of PPDA (Passes per Defensive Action), and it's used to measure how well a team is able to resist pressure from the opposition. 
                    
                    While PPDA measures how aggressively a team presses the opponent, OPPDA measures how many passes a team allows their opponent to make before the opponent successfully performs a defensive action against them.
                    """)
            plot_ppda_oppda_comparison(league_selected)
        
        elif statname == 'Heatmap Correlation':
            plot_correlation_heatmap(league_selected)
            display_comparison(league_selected)
            if compare == 'Yes':
                plot_correlation_heatmap(compare_league_selected)


    # WWWLL
    with tab_list[6]:
        st.header("Why Winners Win and Losers Lose")
        
        statname = st.selectbox('Statistic', ('Top Clubs', 'Statitics', 'Performances', 'Records'), key='wwwll_statname')
        
        if statname != 'Performances':
            league_display_names = list(league_names_mapping.values())
            league_choice = st.selectbox('Choose the league', league_display_names, key='wwwll_choice')
            league_selected = [key for key, value in league_names_mapping.items() if value == league_choice][0]

        if statname == 'Top Clubs':
            analyze_top_4_performance(league_selected)

        elif statname == 'Statitics':
            statname = st.selectbox('Criteria', ('Winners', 'Losers', 'Positions'), key='wi_statname')
            if statname == 'Winners':
                plot_points_comparison(league_selected, statname)
                plot_goals_comparison(league_selected, statname)
                plot_missed_comparison(league_selected, statname)
            elif statname == 'Losers':
                plot_points_comparison(league_selected, statname)
                plot_goals_comparison(league_selected, statname)
                plot_missed_comparison(league_selected, statname)
            elif statname == 'Positions':
                plot_xpts_difference(league_selected)
                plot_xg_difference(league_selected)
                plot_xga_difference(league_selected)

        elif statname == 'Performances':
            perf_criteria = st.selectbox('Criteria', ('Overperformance', 'Underperformance'), key='perf_statname')
            if perf_criteria == 'Overperformance':
                detect_outliers('Overperformance')
            elif perf_criteria == 'Underperformance':
                detect_outliers('Underperformance')

        elif statname == 'Records':
            get_records_antirecords(league_selected)



    # Playing Styles
    with tab_list[7]:
        st.header("Playing Styles")
        league_display_names = list(league_names_mapping_clu.values())
        league_choice = st.selectbox('Choose the league', league_display_names, key='playing_styles_choice')
        league_selected = [key for key, value in league_names_mapping.items() if value == league_choice][0]
        year = st.selectbox('Choose the year', df_u['Year'].unique(), key='playing_styles_year')
        st.write("""
                - Possession refers to the percentage of time a team controls the ball during a match. 
                It is typically calculated as the total time a team has the ball divided by the total match time.
                -  SPPLBP is a metric that measures the ratio of short passes completed to long balls played by a team.
                - npxGD is a metric that reflects a team’s expected goal difference from open play, excluding penalty goals.
                It is calculated as xG - xGA
                """)
        plot_team_clusters_by_year(league_selected, year)
        selected_cluster = st.selectbox("Select a cluster number:", options=list(cluster_descriptions.keys()))
        st.markdown(cluster_descriptions[selected_cluster])

