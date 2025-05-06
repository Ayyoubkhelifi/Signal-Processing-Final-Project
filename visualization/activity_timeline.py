import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

def generate_activity_timeline(y_true, y_pred, class_names, window_size=100):
    """Generate a timeline visualization as base64 image."""
    # If we have more samples than window_size, take the last window_size samples
    if len(y_true) > window_size:
        y_true = y_true[-window_size:]
        y_pred = y_pred[-window_size:]
    
    # Create figure
    plt.figure(figsize=(15, 6))
    
    # Plot true activities
    plt.subplot(2, 1, 1)
    plt.plot(y_true, 'b-', linewidth=2)
    plt.yticks(range(len(class_names)), class_names)
    plt.title('True Activities')
    plt.grid(True)
    
    # Plot predicted activities
    plt.subplot(2, 1, 2)
    plt.plot(y_pred, 'r-', linewidth=2)
    plt.yticks(range(len(class_names)), class_names)
    plt.title('Predicted Activities')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Convert plot to base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_str

# Create an alias for the function to match the import in app.py
plot_activity_timeline = generate_activity_timeline