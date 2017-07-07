import numpy as np
import pandas as pd
import pickle, bz2, ConfigParser
from jeopardy_model.buildModel import *
import plotly as py
import plotly.graph_objs as go


# to do --
# default/initial date range?
# make subplots.
# host and connect

config = ConfigParser.ConfigParser()
config.read('{0}/.python_keys.conf'.format(os.path.expanduser("~")))
py.tools.set_credentials_file(username=config.get('plotly','un'), api_key=config.get('plotly','pw'))

file = "{}/jeopardy_model/data/raw.data".format(os.path.expanduser("~"))
df = pd.read_csv(file,delimiter=',',header=None)
df.columns = ['g', 'gameNumber', 'date', 'winningDays', 'winningDollars', 'winner', 'gender', 'age', 'name', 'career', 'location']

with bz2.BZ2File("/home/jason/jeopardy_model/model_pickles/model.pickle") as f: model = pickle.load(f)
with bz2.BZ2File("/home/jason/jeopardy_model/model_pickles/scaler.pickle") as f: scaler = pickle.load(f)

df2 = constructFeatures(df)
X, X_scaled, Y, scaler, X_fix = processData(df2, None,scaler,None)
df2['probability'] = predict_all(X_scaled,model)
df2 = df2.sort_values(by='date')


fig = py.tools.make_subplots(rows=2, cols=1,subplot_titles=('Model Probability by Previous Wins', 'Zoomable Model Probability Timeseries'))


### box plot
data1 = go.Box(y=df2['probability'],x=df2['prevWins_capped'],name='Boxplot')
#layout1 = go.Layout(yaxis=dict(title='Model Probability',zeroline=False),xaxis=dict(title='Previous Wins'),boxmode='group')
#fig = go.Figure(data=[data1], layout=layout1)
#plot_url = py.plotly.plot(fig, filename='jeopardy_days_box',sharing='public')

### scatterplot
#data = [go.Scatter(y=df2['probability'],x=df2['winningDollars'],text=df2['name'], mode='markers', marker = dict(size='16',color=df2['prevWins_capped'],colorscale='Viridis',showscale=True))]
#layout = go.Layout(yaxis=dict(title='Model Probability',zeroline=False),xaxis=dict(title='Winnings ($)'))
#fig = go.Figure(data=data, layout=layout)
#plot_url = py.plotly.plot(fig, filename='jeopardy_days_box',sharing='public')

### scatterplot by date
data2 = go.Scatter(y=df2['probability'],x=df2['date'],text=df2['name'], mode='lines+markers',
    marker = dict(size='8',
        #colorbar = dict(title = "Previous Wins"),
    color=df2['prevWins_capped'],colorscale='Viridis',showscale=False),name='Timeseries')
    
#layout2 = go.Layout(title='Zoomable Model Probability Timeseries', yaxis=dict(title='Model Probability',zeroline=False),    xaxis=dict(title='Winnings ($)'))
#fig = go.Figure(data=[data2], layout=layout2)
#plot_url = py.plotly.plot(fig, filename='jeopardy_days_box',sharing='public')

fig.append_trace(data1,1,1)
fig.append_trace(data2,2,1)

fig['layout']['xaxis1'].update(title='Previous Wins')
fig['layout']['yaxis1'].update(title='Model Probability',zeroline=False)
#fig['layout']['boxmode1'].update(group)

fig['layout']['xaxis2'].update(title='Game Date')
fig['layout']['yaxis2'].update(title='Model Probability',zeroline=False)


plot_url = py.plotly.plot(fig, filename='jeopardy_model_plotly',sharing='public')    
    

