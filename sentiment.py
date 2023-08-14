import plotly.graph_objects as go

def create_bullet_chart(actual, reference, target):
    fig = go.Figure()

    # Reference bar (e.g., average sentiment)
    fig.add_trace(go.Bar(
        y=['Sentiment'],
        x=[reference],
        orientation='h',
        name='Reference',
        marker=dict(color='gray')
    ))

    # Actual bar (e.g., actual sentiment score)
    fig.add_trace(go.Bar(
        y=['Sentiment'],
        x=[actual],
        orientation='h',
        name='Actual',
        marker=dict(color='blue')
    ))

    # Target marker (e.g., target sentiment score)
    fig.add_layout_image(
        dict(
            source="https://raw.githubusercontent.com/plotly/plotly.js/master/test/image/baseline/indicator_bullet.png",
            xref="x",
            yref="y",
            x=target,
            y='Sentiment',
            sizex=5,
            sizey=0.5,
            xanchor="center",
            yanchor="middle"
        )
    )

    fig.update_layout(barmode='overlay', showlegend=False)
    return fig
