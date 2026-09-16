import matplotlib
matplotlib.use("Agg")

import base64
from io import BytesIO

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

# DATA LOADING AND PREPARATION

df = pd.read_csv('data/superstore_cleaned.csv',encoding='latin1')

# Clean column names
df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace(r"[^\w]", "", regex=True)
)

# Convert dates
for col in ["order_date", "ship_date"]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

# Convert numeric columns
for col in ["sales", "quantity", "discount", "profit", "postal_code"]:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Derived columns
if "ship_date" in df.columns and "order_date" in df.columns:
    df["shipping_days"] = (df["ship_date"] - df["order_date"]).dt.days

if "sales" in df.columns and "profit" in df.columns:
    df["profit_margin"] = np.where(
        df["sales"] != 0,
        (df["profit"] / df["sales"]) * 100,
        np.nan
    )

if "order_date" in df.columns:
    df["year_month"] = df["order_date"].dt.to_period("M").astype(str)

# For time-based charts
df_time = df.dropna(subset=["order_date"]).copy() if "order_date" in df.columns else df.copy()

# STATIC IMAGE HELPERS

sns.set_style("whitegrid")


def fig_to_base64(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buffer.seek(0)
    return "data:image/png;base64," + base64.b64encode(buffer.read()).decode()


def make_profit_distribution_image(dataframe):
    plot_df = dataframe.dropna(subset=["profit"]).copy()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(plot_df["profit"], kde=True, ax=ax)
    ax.axvline(0, linestyle="--", linewidth=2, color="black")
    ax.set_title("Profit Distribution")
    ax.set_xlabel("Profit")
    ax.set_ylabel("Frequency")
    return fig_to_base64(fig)


def make_subcategory_profit_image(dataframe):
    plot_df = dataframe.dropna(subset=["subcategory", "profit"]).copy()

    subcat_profit = (
        plot_df.groupby("subcategory")["profit"]
        .mean()
        .sort_values()
        .reset_index()
    )

    subcat_profit["subcategory"] = subcat_profit["subcategory"].astype(str)

    colors = ["red" if x < 0 else "steelblue" for x in subcat_profit["profit"]]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(
        subcat_profit["subcategory"].tolist(),
        subcat_profit["profit"].tolist(),
        color=colors
    )
    ax.set_title("Average Profit by Sub-Category")
    ax.set_xlabel("Average Profit")
    ax.set_ylabel("Sub-Category")
    return fig_to_base64(fig)


def make_corr_image(dataframe):
    corr_cols = ["sales", "quantity", "discount", "profit"]
    plot_df = dataframe[corr_cols].apply(pd.to_numeric, errors="coerce")

    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(plot_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    return fig_to_base64(fig)


profit_dist_img = make_profit_distribution_image(df)
subcat_profit_img = make_subcategory_profit_image(df)
corr_img = make_corr_image(df)

# DASH APP
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "SuperStore Dashboard"

app.layout = dbc.Container(
    fluid=True,
    children=[
        dbc.Row(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H1("SuperStore Performance & Profitability Dashboard", className="mb-2"),
                            html.Div("Student Number: 402309260", className="text-muted"),
                            html.P(
                                "Interactive dashboard analysing sales, profit, discount impact, regional performance, shipping efficiency, and customer segments.",
                                className="mt-2 mb-0",
                            ),
                        ]
                    ),
                    className="shadow-sm mb-4",
                ),
                width=12,
            )
        ),

        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Filters", className="card-title"),

                                html.Label("Region"),
                                dcc.Dropdown(
                                    id="region_filter",
                                    options=[{"label": r, "value": r} for r in sorted(df["region"].dropna().unique())],
                                    multi=True,
                                    placeholder="Select region(s)",
                                ),
                                html.Br(),

                                html.Label("Category"),
                                dcc.Dropdown(
                                    id="category_filter",
                                    options=[{"label": c, "value": c} for c in sorted(df["category"].dropna().unique())],
                                    multi=True,
                                    placeholder="Select category(s)",
                                ),
                                html.Br(),

                                html.Label("Segment"),
                                dcc.Dropdown(
                                    id="segment_filter",
                                    options=[{"label": s, "value": s} for s in sorted(df["segment"].dropna().unique())],
                                    multi=True,
                                    placeholder="Select segment(s)",
                                ),
                                html.Br(),

                                html.Label("Date Range"),
                                dcc.DatePickerRange(
                                    id="date_filter",
                                    min_date_allowed=df_time["order_date"].min() if "order_date" in df_time.columns else None,
                                    max_date_allowed=df_time["order_date"].max() if "order_date" in df_time.columns else None,
                                    start_date=df_time["order_date"].min() if "order_date" in df_time.columns else None,
                                    end_date=df_time["order_date"].max() if "order_date" in df_time.columns else None,
                                    display_format="YYYY-MM-DD",
                                ),
                                html.Br(),
                                html.Br(),

                                html.Label("Regional Chart Metric"),
                                dbc.RadioItems(
                                    id="metric_toggle",
                                    options=[
                                        {"label": " Sales", "value": "sales"},
                                        {"label": " Profit", "value": "profit"},
                                    ],
                                    value="sales",
                                    inline=True,
                                ),
                            ]
                        ),
                        className="shadow-sm mb-4",
                    ),
                    md=3,
                ),

                dbc.Col(
                    [
                        dbc.Row(
                            [
                                dbc.Col(dbc.Card(dbc.CardBody([html.H6("Total Sales"), html.H3(id="kpi_sales")])), md=3),
                                dbc.Col(dbc.Card(dbc.CardBody([html.H6("Total Profit"), html.H3(id="kpi_profit")])), md=3),
                                dbc.Col(dbc.Card(dbc.CardBody([html.H6("Average Profit Margin"), html.H3(id="kpi_margin")])), md=3),
                                dbc.Col(dbc.Card(dbc.CardBody([html.H6("Number of Orders"), html.H3(id="kpi_orders")])), md=3),
                            ],
                            className="g-3 mb-4",
                        ),

                        dbc.Tabs(
                            [
                                dbc.Tab(
                                    label="Overview",
                                    children=[
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            [
                                                                html.H5("Sales vs Profit Scatter Plot"),
                                                                dcc.Graph(id="scatter_chart"),
                                                                html.P(
                                                                    "This chart shows the relationship between sales and profit. Marker size reflects quantity and colour reflects customer segment.",
                                                                    className="mt-2",
                                                                ),
                                                            ]
                                                        ),
                                                        className="shadow-sm mb-4",
                                                    ),
                                                    md=12,
                                                )
                                            ]
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            [
                                                                html.H5("Monthly Trend"),
                                                                dcc.Graph(id="time_chart"),
                                                                html.P(
                                                                    "The time series shows sales over time with an interactive range slider and preset buttons.",
                                                                    className="mt-2",
                                                                ),
                                                            ]
                                                        ),
                                                        className="shadow-sm mb-4",
                                                    ),
                                                    md=12,
                                                )
                                            ]
                                        ),
                                    ],
                                ),

                                dbc.Tab(
                                    label="Regional Analysis",
                                    children=[
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            [
                                                                html.H5("Regional Performance"),
                                                                dcc.Graph(id="region_chart"),
                                                                html.P(
                                                                    "This chart compares regional performance using the selected metric from the radio buttons.",
                                                                    className="mt-2",
                                                                ),
                                                            ]
                                                        ),
                                                        className="shadow-sm mb-4",
                                                    ),
                                                    md=8,
                                                ),
                                                dbc.Col(
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            [
                                                                html.H5("Shipping Efficiency"),
                                                                dcc.Graph(id="shipping_chart"),
                                                                html.P(
                                                                    "Average shipping days by ship mode help identify delivery efficiency differences.",
                                                                    className="mt-2",
                                                                ),
                                                            ]
                                                        ),
                                                        className="shadow-sm mb-4",
                                                    ),
                                                    md=4,
                                                ),
                                            ]
                                        )
                                    ],
                                ),

                                dbc.Tab(
                                    label="Product & Segment Analysis",
                                    children=[
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            [
                                                                html.H5("Segment Performance"),
                                                                dcc.Graph(id="segment_chart"),
                                                                html.P(
                                                                    "This chart compares total sales and total profit across customer segments.",
                                                                    className="mt-2",
                                                                ),
                                                            ]
                                                        ),
                                                        className="shadow-sm mb-4",
                                                    ),
                                                    md=12,
                                                )
                                            ]
                                        )
                                    ],
                                ),

                                dbc.Tab(
                                    label="Static Visualisations",
                                    children=[
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            [
                                                                html.H5("Profit Distribution"),
                                                                html.Img(src=profit_dist_img, style={"width": "100%"}),
                                                                html.P(
                                                                    "The histogram shows the distribution of profit values. The dashed vertical line marks the break-even point at profit = 0.",
                                                                    className="mt-2",
                                                                ),
                                                            ]
                                                        ),
                                                        className="shadow-sm mb-4",
                                                    ),
                                                    md=6,
                                                ),
                                                dbc.Col(
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            [
                                                                html.H5("Average Profit by Sub-Category"),
                                                                html.Img(src=subcat_profit_img, style={"width": "100%"}),
                                                                html.P(
                                                                    "Negative average profit values are highlighted in red to draw attention to weaker product lines.",
                                                                    className="mt-2",
                                                                ),
                                                            ]
                                                        ),
                                                        className="shadow-sm mb-4",
                                                    ),
                                                    md=6,
                                                ),
                                            ]
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            [
                                                                html.H5("Correlation Heatmap"),
                                                                html.Img(src=corr_img, style={"width": "100%"}),
                                                                html.P(
                                                                    "The heatmap summarises the relationships between sales, quantity, discount, and profit.",
                                                                    className="mt-2",
                                                                ),
                                                            ]
                                                        ),
                                                        className="shadow-sm mb-4",
                                                    ),
                                                    md=12,
                                                )
                                            ]
                                        ),
                                    ],
                                ),
                            ]
                        ),
                    ],
                    md=9,
                ),
            ]
        ),
    ],
    style={"padding": "20px"},
)

# CALLBACK

@app.callback(
    [
        Output("kpi_sales", "children"),
        Output("kpi_profit", "children"),
        Output("kpi_margin", "children"),
        Output("kpi_orders", "children"),
        Output("scatter_chart", "figure"),
        Output("time_chart", "figure"),
        Output("region_chart", "figure"),
        Output("shipping_chart", "figure"),
        Output("segment_chart", "figure"),
    ],
    [
        Input("region_filter", "value"),
        Input("category_filter", "value"),
        Input("segment_filter", "value"),
        Input("date_filter", "start_date"),
        Input("date_filter", "end_date"),
        Input("metric_toggle", "value"),
    ],
)
def update_dashboard(selected_regions, selected_categories, selected_segments, start_date, end_date, metric):
    """
    Update KPIs and interactive charts based on filter selections.
    """
    filtered = df.copy()

    if selected_regions:
        filtered = filtered[filtered["region"].isin(selected_regions)]

    if selected_categories:
        filtered = filtered[filtered["category"].isin(selected_categories)]

    if selected_segments:
        filtered = filtered[filtered["segment"].isin(selected_segments)]

    if "order_date" in filtered.columns:
        if start_date:
            filtered = filtered[filtered["order_date"] >= pd.to_datetime(start_date)]
        if end_date:
            filtered = filtered[filtered["order_date"] <= pd.to_datetime(end_date)]

    if filtered.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="No data available for selected filters")
        return "0", "0", "0%", "0", empty_fig, empty_fig, empty_fig, empty_fig, empty_fig

    total_sales = filtered["sales"].sum()
    total_profit = filtered["profit"].sum()
    avg_margin = filtered["profit_margin"].mean()
    num_orders = filtered["order_id"].nunique() if "order_id" in filtered.columns else len(filtered)

    scatter_fig = px.scatter(
        filtered,
        x="sales",
        y="profit",
        color="segment",
        size="quantity",
        hover_data=["product_name", "category", "subcategory", "discount", "region"],
        title="Sales vs Profit",
    )
    scatter_fig.update_layout(legend_title_text="Segment")

    monthly = (
        filtered.dropna(subset=["order_date"])
        .assign(year_month=filtered.dropna(subset=["order_date"])["order_date"].dt.to_period("M").astype(str))
        .groupby("year_month", as_index=False)["sales"]
        .sum()
    )

    time_fig = px.line(monthly, x="year_month", y="sales", markers=True, title="Monthly Sales Trend")
    time_fig.update_layout(
        xaxis_title="Year-Month",
        yaxis_title="Sales",
        xaxis=dict(rangeslider=dict(visible=True)),
    )

    region_summary = filtered.groupby("region", as_index=False)[metric].sum()
    region_fig = px.bar(
        region_summary,
        x="region",
        y=metric,
        title=f"Regional {metric.title()} Comparison",
        text_auto=".2s",
    )

    if "shipping_days" in filtered.columns and "ship_mode" in filtered.columns:
        shipping_summary = filtered.groupby("ship_mode", as_index=False)["shipping_days"].mean()
        shipping_fig = px.bar(
            shipping_summary,
            x="ship_mode",
            y="shipping_days",
            title="Average Shipping Days by Ship Mode",
            text_auto=".2f",
        )
    else:
        shipping_fig = go.Figure()
        shipping_fig.update_layout(title="Shipping data not available")

    segment_summary = filtered.groupby("segment", as_index=False)[["sales", "profit"]].sum()
    segment_fig = go.Figure()
    segment_fig.add_trace(go.Bar(x=segment_summary["segment"], y=segment_summary["sales"], name="Sales"))
    segment_fig.add_trace(go.Bar(x=segment_summary["segment"], y=segment_summary["profit"], name="Profit"))
    segment_fig.update_layout(
        title="Sales and Profit by Segment",
        barmode="group",
        xaxis_title="Segment",
        yaxis_title="Value",
    )

    return (
        f"${total_sales:,.2f}",
        f"${total_profit:,.2f}",
        f"{avg_margin:,.2f}%",
        f"{num_orders:,}",
        scatter_fig,
        time_fig,
        region_fig,
        shipping_fig,
        segment_fig,
    )


if __name__ == "__main__":
    app.run(debug=True)