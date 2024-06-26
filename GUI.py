import dash
from dash import dcc, html  # Updated import statements
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
from scipy.stats import zscore  # Corrected import
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

data = pd.read_excel('Dataset.xlsx')

# Data Cleaning
data.drop_duplicates(inplace=True)
imputer = SimpleImputer(strategy='mean')
data.iloc[:, :-1] = imputer.fit_transform(data.iloc[:, :-1])

# Removing outliers
z_scores = np.abs(data.iloc[:, :-1].apply(zscore))  # Corrected import usage
data = data[(z_scores < 3).all(axis=1)]

# Data Transformation
scaler = StandardScaler()
data.iloc[:, :-1] = scaler.fit_transform(data.iloc[:, :-1])

# Splitting data
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Initialize Dash app
app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    html.H1("Water Quality Predictive Dashboard"),
    dcc.Graph(id='feature-importance-graph'),
    html.Label([
        "Select a feature to display",
        dcc.Dropdown(
            id='feature-dropdown',
            options=[{'label': feature, 'value': feature} for feature in X.columns],  # Corrected variable name
            value=X.columns[0]  # Default to first column
        )
    ]),
    html.Button('Update Graph', id='submit-button', n_clicks=0),
])

@app.callback(
    Output('feature-importance-graph', 'figure'),
    [Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('feature-dropdown', 'value')]
)
def update_graph(n_clicks, selected_feature):
    fig = px.histogram(data, x=selected_feature)
    return fig

# Run app
if __name__ == '__main__':
    app.run_server(debug=True)
