"""
Simple KNN vs SVM comparison diagram showing face data points and decision boundaries.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap


def create_knn_vs_svm_diagram():
    """Create a simple comparison showing face points, boundaries, and predictions."""
    
    # Generate sample face data (2D projection of 128-dim embeddings)
    np.random.seed(42)
    n_samples_per_class = 20
    
    # 7 classes from your dataset
    classes = ['Jennifer', 'Unknown', 'Steven', 'Edbert', 'Justin', 'Angel', 'Mario']
    colors = ['#2196F3', '#9E9E9E', '#F44336', '#4CAF50', '#FF9800', '#9C27B0', '#00BCD4']
    
    # Generate data points in 2D space (arranged in a circle pattern for 7 classes)
    angles = np.linspace(0, 2*np.pi, 8)[:-1]  # 7 angles for 7 people
    radius = 2.5
    centers = [(radius * np.cos(angle), radius * np.sin(angle)) for angle in angles]
    
    X = []
    y = []
    
    for i, (center, color) in enumerate(zip(centers, colors)):
        x_points = np.random.randn(n_samples_per_class) * 0.8 + center[0]
        y_points = np.random.randn(n_samples_per_class) * 0.8 + center[1]
        X.extend(list(zip(x_points, y_points)))
        y.extend([i] * n_samples_per_class)
    
    X = np.array(X)
    y = np.array(y)
    
    # Create mesh for decision boundaries
    h = 0.1  # step size in mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Test point to show prediction
    test_point = np.array([[0.5, 1.0]])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # ========== LEFT: KNN ==========
    # Train KNN
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn.fit(X, y)
    
    # Predict mesh for KNN
    Z_knn = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_knn = Z_knn.reshape(xx.shape)
    
    # Plot decision regions (simple filled contours)
    # Use a colormap that can handle 7 classes
    cmap_light = ListedColormap(['#E3F2FD', '#F3E5F5', '#FFEBEE', '#E8F5E9', '#FFF3E0', '#F1F8E9', '#E0F2F1'])
    ax1.contourf(xx, yy, Z_knn, alpha=0.4, cmap=cmap_light, levels=len(classes))
    
    # Plot training points
    for i, (cls, color) in enumerate(zip(classes, colors)):
        mask = y == i
        ax1.scatter(X[mask, 0], X[mask, 1], c=color, s=60, 
                   edgecolors='black', linewidths=1.5, 
                   label=cls, zorder=3)
    
    # Plot test point
    test_pred_knn = knn.predict(test_point)[0]
    test_color = colors[test_pred_knn]
    ax1.scatter(test_point[0, 0], test_point[0, 1], 
               c=test_color, s=200, marker='*', 
               edgecolors='black', linewidths=2, 
               label=f'Test point → {classes[test_pred_knn]}', 
               zorder=5)
    
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_xlabel('Feature Dimension 1', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Feature Dimension 2', fontsize=12, fontweight='bold')
    ax1.set_title('KNN Classifier\n(Irregular Boundaries, Finds 5 Nearest Neighbors)', 
                 fontsize=14, fontweight='bold', color='#D32F2F')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=9, ncol=2)
    
    # ========== RIGHT: SVM ==========
    # Train SVM
    svm = SVC(kernel='linear', probability=True, class_weight='balanced')
    svm.fit(X, y)
    
    # Predict mesh for SVM
    Z_svm = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_svm = Z_svm.reshape(xx.shape)
    
    # Plot decision regions
    ax2.contourf(xx, yy, Z_svm, alpha=0.4, cmap=cmap_light, levels=len(classes))
    
    # Draw linear decision boundaries for SVM (one-vs-rest)
    # For each class, draw the decision boundary as a line
    if hasattr(svm, 'coef_'):
        x_line = np.linspace(x_min, x_max, 100)
        y_line = np.linspace(y_min, y_max, 100)
        
        # Draw boundaries between all pairs of classes
        for i in range(len(classes)):
            if i < len(svm.coef_):
                w = svm.coef_[i]
                b = svm.intercept_[i]
                
                # Draw line: w[0]*x + w[1]*y + b = 0
                # y = (-w[0]*x - b) / w[1]
                if abs(w[1]) > 1e-6:  # Avoid division by zero
                    y_boundary = (-w[0] * x_line - b) / w[1]
                    # Only plot within the visible range
                    mask = (y_boundary >= y_min) & (y_boundary <= y_max)
                    if np.any(mask):
                        ax2.plot(x_line[mask], y_boundary[mask], 'k-', 
                               linewidth=2, alpha=0.7, linestyle='--')
    
    # Plot support vectors
    ax2.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1], 
               s=100, facecolors='none', edgecolors='black', 
               linewidths=2, label='Support Vectors', zorder=4)
    
    # Plot training points
    for i, (cls, color) in enumerate(zip(classes, colors)):
        mask = y == i
        ax2.scatter(X[mask, 0], X[mask, 1], c=color, s=60, 
                   edgecolors='black', linewidths=1.5, 
                   label=cls, zorder=3)
    
    # Plot test point
    test_pred_svm = svm.predict(test_point)[0]
    test_color_svm = colors[test_pred_svm]
    ax2.scatter(test_point[0, 0], test_point[0, 1], 
               c=test_color_svm, s=200, marker='*', 
               edgecolors='black', linewidths=2, 
               label=f'Test point → {classes[test_pred_svm]}', 
               zorder=5)
    
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.set_xlabel('Feature Dimension 1', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Feature Dimension 2', fontsize=12, fontweight='bold')
    ax2.set_title('SVM Classifier\n(Linear Decision Boundaries, Uses Support Vectors)', 
                 fontsize=14, fontweight='bold', color='#388E3C')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=9, ncol=2)
    
    # Overall title
    fig.suptitle('KNN vs SVM: Face Recognition Classification', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


if __name__ == '__main__':
    print("Creating KNN vs SVM diagram with face data points...")
    fig = create_knn_vs_svm_diagram()
    fig.savefig('knn_vs_svm.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("Saved: knn_vs_svm.png")
    
    plt.show()
