import marimo

__generated_with = "0.14.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from polars import read_avro
    import marimo as mo

    df = read_avro("processed.avro")
    return df, mo


@app.cell
def _(df, mo):
    item_length = mo.ui.slider(start=1, stop=len(df), label="item count")
    item_length
    return (item_length,)


@app.cell
def _(df, item_length):
    import plotly.express as px
    import polars as pl

    # Create derived numerical features for scatter plot
    _scatter_df = df.head(item_length.value).with_columns([
        pl.col("content").str.len_chars().alias("content_length"),
        pl.col("author_name").str.len_chars().alias("author_name_length"),
        pl.col("is_bot_comment").cast(pl.String).alias("bot_status")
    ])

    # Create scatter plot
    scatter_fig = px.scatter(
        _scatter_df.to_pandas(),
        x="content_length",
        y="author_name_length", 
        color="bot_status",
        hover_data=["author_name", "video_id"],
        title=f"Comment Content Length vs Author Name Length (First {item_length.value} items)",
        labels={
            "content_length": "Comment Content Length (characters)",
            "author_name_length": "Author Name Length (characters)",
            "bot_status": "Bot Comment"
        },
        color_discrete_map={"True": "red", "False": "blue"}
    )

    scatter_fig.update_layout(
        width=800,
        height=600,
        showlegend=True
    )

    scatter_fig
    return


if __name__ == "__main__":
    app.run()
