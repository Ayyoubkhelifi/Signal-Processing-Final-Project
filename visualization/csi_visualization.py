import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from matplotlib.colors import ListedColormap

def generate_csi_visualization(csi_data, activities, class_names, n_samples=1000):
    """Generate a CSI visualization with activity labels as base64 image."""
    # If we have more samples than n_samples, take the last n_samples
    if csi_data.shape[0] > n_samples:
        csi_data = csi_data[-n_samples:]
        activities = activities[-n_samples:]
    
    # Create a colormap for activities
    n_classes = len(class_names)
    colors = plt.cm.jet(np.linspace(0, 1, n_classes))
    activity_cmap = ListedColormap(colors)
    
    # Create figure
    plt.figure(figsize=(15, 8))
    
    # Plot CSI data (first 3 principal components)
    for i in range(min(3, csi_data.shape[1])):
        plt.subplot(4, 1, i+1)
        plt.plot(csi_data[:, i], 'k-', alpha=0.7)
        plt.title(f'CSI Principal Component {i+1}')
        plt.grid(True)
    
    # Plot activities
    plt.subplot(4, 1, 4)
    plt.scatter(range(len(activities)), activities, c=activities, cmap=activity_cmap, 
                marker='|', s=100, alpha=0.8)
    plt.yticks(range(len(class_names)), class_names)
    plt.title('Activities')
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
plot_csi_with_activities = generate_csi_visualization