import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import io
import base64

def generate_confusion_matrix(y_true, y_pred, class_names):
    """Generate a confusion matrix visualization as base64 image."""
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize the confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Activity')
    plt.xlabel('Predicted Activity')
    plt.title('Confusion Matrix for Human Activity Recognition')
    
    # Convert plot to base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_str

# Create an alias for the function to match the import in app.py
plot_confusion_matrix = generate_confusion_matrix