import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

# Load data
latent_vectors = pd.read_csv('./data/processed/latent_vectors_anonymized.csv')
rlps_data = pd.read_csv('./data/processed/rlps_2023_data_anonymized.csv')
latent_vectors.rename(columns={'ANONYMIZED_BOXID': 'BOXID'}, inplace=True)
latent_vectors.rename(columns={'ANONYMIZED_CITY': 'GEMEENTE'}, inplace=True)

rlps_data.rename(columns={'ANONYMIZED_BOXID': 'BOXID'}, inplace=True)
rlps_data.rename(columns={'ANONYMIZED_CITY': 'GEMEENTE'}, inplace=True)


# Get unique values for dropdowns
gemeente_options = latent_vectors['GEMEENTE'].unique()
month_options = sorted(latent_vectors['MONTH'].unique())

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = dbc.Container([
    html.H1("3D Scatter & Load Profile Dashboard", className="text-center mb-4"),

    # Row for dropdowns
    dbc.Row([
        dbc.Col([html.Label("GEMEENTE:"),
                 dcc.Dropdown(
                     id='gemeente-dropdown',
                     options=[{'label': g, 'value': g} for g in gemeente_options],
                     value=gemeente_options[0],  # Default to first option
                     clearable=False)], width=4),

        dbc.Col([html.Label("MONTH:"),
                 dcc.Dropdown(
                     id='month-dropdown',
                     options=[{'label': str(m), 'value': m} for m in month_options],
                     value=month_options[0],  # Default to first option
                     clearable=False)], width=4),

        dbc.Col([html.Label("BOXID (Optional):"),
                 dcc.Dropdown(id='highlight-dropdown', clearable=True, placeholder="Select a BOXID")], width=4)
    ], className="mb-3"),

    dcc.Graph(id='scatter-plot')
], fluid=True)


@app.callback(
    Output('highlight-dropdown', 'options'),
    Input('gemeente-dropdown', 'value'),
    Input('month-dropdown', 'value')
)
def update_highlight_options(selected_gemeente, selected_month):
    filtered_data = latent_vectors[(latent_vectors['GEMEENTE'] == selected_gemeente) &
                                   (latent_vectors['MONTH'] == selected_month)]
    return [{'label': boxid, 'value': boxid} for boxid in filtered_data['BOXID'].unique()]


@app.callback(
    Output('scatter-plot', 'figure'),
    Input('gemeente-dropdown', 'value'),
    Input('month-dropdown', 'value'),
    Input('highlight-dropdown', 'value')
)
def update_plot(selected_gemeente, selected_month, highlight_boxid):
    models = ['isomap', 'sphere', 'umap', 'autoencoder']
    titles = [f"Model: {model.capitalize()}" for model in models]

    # Define colors
    cluster_colors = {i: plt.get_cmap("tab10")(i) for i in range(4)}

    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=titles + ["Load Profile", ""],
        specs=[
            [{'type': 'scatter3d'}, {'type': 'scatter3d'}],
            [{'type': 'scatter3d'}, {'type': 'scatter3d'}],
            [{'type': 'xy', 'colspan': 2}, None]
        ],
        row_heights=[0.4, 0.4, 0.2]
    )

    for i, model in enumerate(models):
        row = i // 2 + 1
        col = i % 2 + 1

        data = latent_vectors[(latent_vectors['GEMEENTE'] == selected_gemeente) &
                              (latent_vectors['MONTH'] == selected_month) &
                              (latent_vectors['MODEL'] == model)]
        data_highlight = data[data['BOXID'] == highlight_boxid]

        if row == 1 and col == 1:
            showlegend = True
        else:
            showlegend = False

        for cluster_id in sorted(data['CLUSTER'].unique()):
            cluster_data = data[data['CLUSTER'] == cluster_id]
            fig.add_trace(
                go.Scatter3d(
                    x=cluster_data['Z1'], y=cluster_data['Z2'], z=cluster_data['Z3'],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=f"rgb{tuple(int(c * 255) for c in cluster_colors[cluster_id][:3])}",
                        opacity=0.8
                    ),
                    text=cluster_data['BOXID'],
                    showlegend=False
                ),
                row=row, col=col
            )

            fig.add_trace(
                go.Scatter3d(
                    x=[None], y=[None], z=[None],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color=f"rgb{tuple(int(c * 255) for c in cluster_colors[cluster_id][:3])}"
                    ),
                    name=f"Cluster {cluster_id}",
                    showlegend=showlegend
                )
            )

        if highlight_boxid:
            fig.add_trace(
                go.Scatter3d(
                    x=data_highlight['Z1'], y=data_highlight['Z2'], z=data_highlight['Z3'],
                    mode='markers',
                    marker=dict(size=5, color="purple"),
                    name=f"Highlight ({highlight_boxid})",
                    showlegend=False
                ),
                row=row, col=col
            )

    # Add time-series if highlight is selected
    if highlight_boxid:
        rlps = rlps_data[(rlps_data['GEMEENTE'] == selected_gemeente) &
                         (rlps_data['MONTH'] == selected_month) &
                         (rlps_data['BOXID'] == highlight_boxid)].drop(
            columns=["GEMEENTE", "MONTH", "BOXID", "CLUSTER"])
        date_range = pd.date_range(start=f"2023-{selected_month:02d}-01", periods=96, freq='15min')
        rlps_df = pd.DataFrame(rlps.values.flatten(), index=date_range, columns=['power'])

        fig.add_trace(
            go.Scatter(
                x=rlps_df.index, y=rlps_df['power'],
                mode='lines',
                line=dict(color="purple", width=2),
                name="Load Profile"
            ),
            row=3, col=1
        )

    fig.update_layout(
        title="3D Scatter Plots for Different Models & Load Profile",
        height=1200, width=1100,
        showlegend=True
    )
    return fig


if __name__ == '__main__':
    app.run(debug=False, port=8050, host="0.0.0.0")
