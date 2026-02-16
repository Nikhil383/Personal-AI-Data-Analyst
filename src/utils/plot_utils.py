import matplotlib.pyplot as plt
import io
import uuid
import os
from pathlib import Path

def get_plot_as_image(fig=None):
    """
    Converts a matplotlib figure to a bytes buffer (PNG) for Streamlit to display.
    If no figure is provided, uses the current active figure (plt.gcf()).
    """
    if fig is None:
        fig = plt.gcf()
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)  # Close to prevent memory leaks
    return buf

def save_plot_to_file(destination_dir="static/plots", fig=None):
    """
    Saves the current or provided figure to a file and returns the path.
    Useful if we want to persist plots.
    """
    if fig is None:
        fig = plt.gcf()
        
    os.makedirs(destination_dir, exist_ok=True)
    filename = f"plot_{uuid.uuid4().hex}.png"
    filepath = os.path.join(destination_dir, filename)
    
    fig.savefig(filepath, format="png", bbox_inches='tight')
    plt.close(fig)
    return filepath
