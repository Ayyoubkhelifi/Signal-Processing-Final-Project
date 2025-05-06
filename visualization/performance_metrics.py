import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import io
import base64

def plot_performance_metrics(y_true, y_pred, class_names):
    """
    Visualize model performance metrics for each activity class.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of activity names
        
    Returns:
        HTML img tag with the performance metrics visualization
    """
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    accuracy = accuracy_score(y_true, y_pred)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Set width of bars
    barWidth = 0.25
    
    # Set positions of bars on X axis
    r1 = np.arange(len(class_names))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    
    # Create bars
    plt.bar(r1, precision, width=barWidth, edgecolor='white', label='Precision')
    plt.bar(r2, recall, width=barWidth, edgecolor='white', label='Recall')
    plt.bar(r3, f1, width=barWidth, edgecolor='white', label='F1-Score')
    
    # Add overall accuracy text
    plt.text(len(class_names)/2, 0.9, f'Overall Accuracy: {accuracy:.2f}', 
             horizontalalignment='center', size=12, bbox=dict(facecolor='white', alpha=0.5))
    
    # Add labels and legend
    plt.xlabel('Activity', fontweight='bold')
    plt.ylabel('Score', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(class_names))], class_names, rotation=45)
    plt.ylim(0, 1)
    plt.legend()
    plt.title('Model Performance Metrics by Activity Class')
    
    plt.tight_layout()
    
    # Convert plot to base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return f'<img src="data:image/png;base64,{img_str}" alt="Performance Metrics">'