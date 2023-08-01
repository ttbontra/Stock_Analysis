import matplotlib.pyplot as plt
import io
import base64

def generate_plot():
    # Your data processing and plot generation code here
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return plot_url
