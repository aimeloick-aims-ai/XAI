import numpy as np
import optuna
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, mean_squared_error

def get_surrogate_model_coefficients(X, y, weight, model="Ridge", mode="Regression",n_trials=30, random_state=42):
    """
    Here we fit a weighted surrogate model using Optuna for hyperparameter optimization.
    Returns intercept and coefficients.
    """
    if np.sum(weight) == 0:
        print("Warning: All weights are zero (increase kernel_width or number_sample)")

    def objective(trial):
        #  Regression Part 
        if mode == "Regression" and model == "Ridge":
            alpha = trial.suggest_float("alpha", 1e-4, 100.0, log=True)
            surrogate = Ridge(alpha=alpha)
        elif mode == "Regression" and model == "LR":
            surrogate = LinearRegression()
        else:
            raise ValueError("Invalid model or mode")

        surrogate.fit(X, y, sample_weight=weight)
        y_pred = surrogate.predict(X)
        if mode == "Regression":
            loss = mean_squared_error(y, y_pred, sample_weight=weight)
        else:
            loss = 1 - accuracy_score(y, y_pred)
        return loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = study.best_params

    if mode == "Regression" and model == "Ridge":
        surrogate = Ridge(alpha=best_params["alpha"])
    elif mode == "Regression" and model == "LR":
        surrogate = LinearRegression()

    surrogate.fit(X, y, sample_weight=weight)
    intercept = float(surrogate.intercept_)
    coefficients = surrogate.coef_
    return intercept, coefficients


def My_importance_plot(intercept, coeff, feature_names, feature_values=None, 
                       sort_by_abs=True, method="LIME", title=None):
    """
    Code provide of Claude
    """
    coeff = np.array(coeff)
    feature_names = np.array(feature_names)
    
    if feature_values is not None:
        feature_values = np.array(feature_values)

    # Sorting of contribution
    if sort_by_abs:
        sorted_idx = np.argsort(np.abs(coeff))[::-1]
        coeff = coeff[sorted_idx]
        feature_names = feature_names[sorted_idx]
        if feature_values is not None:
            feature_values = feature_values[sorted_idx]

    # Beautiful colors
    colors = ['#3498DB' if val < 0 else '#E74C3C' for val in coeff]
    edge_colors = ['#2980B9' if val < 0 else '#C0392B' for val in coeff]

    # Create figure
    plt.figure(figsize=(10, 8))
    y_pos = np.arange(len(feature_names))
    
    # Plot bars
    bars = plt.barh(y_pos, coeff, color=colors, edgecolor=edge_colors, 
                    linewidth=1.5, alpha=0.85)
    
    # Add coefficient values on bars
    for i, (bar, val) in enumerate(zip(bars, coeff)):
        width = bar.get_width()
        x_pos = width + (0.02 * (coeff.max() - coeff.min())) if width > 0 else width - (0.02 * (coeff.max() - coeff.min()))
        ha = 'left' if width > 0 else 'right'
        plt.text(x_pos, bar.get_y() + bar.get_height()/2, 
                f'{val:.4f}',
                ha=ha, va='center', fontsize=9, fontweight='bold')
    
    # Y-axis labels: feature names only
    plt.yticks(y_pos, feature_names, fontsize=10)
    
    # Add feature values on the right side if provided
    if feature_values is not None:
        ax = plt.gca()
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(y_pos)
        value_labels = [f"{val:.3f}" for val in feature_values]
        ax2.set_yticklabels(value_labels, fontsize=9, color='gray')
        ax2.set_ylabel("Feature Values", fontsize=11, fontweight='bold', color='gray')
        ax2.invert_yaxis()
    
    plt.xlabel("Contribution to Prediction", fontsize=12, fontweight='bold')
    
    # Add vertical line at x=0
    plt.axvline(x=0, color='black', linestyle='-', linewidth=1.2, alpha=0.7)
    
    # Grid
    plt.grid(True, axis='x', alpha=0.3, linestyle='--')
    
    # Title
    predicted_value = intercept + coeff.sum()
    if title is None:
        method_emoji = "" if method.upper() == "LIME" else "🎯"
        title = f"{method_emoji} {method.upper()} Explanation\n"
        title += f"Predicted = {predicted_value:.4f} (Intercept = {intercept:.4f})"
    plt.title(title, fontsize=13, fontweight='bold', pad=15)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#E74C3C', edgecolor='#C0392B', label='Positive'),
        Patch(facecolor='#3498DB', edgecolor='#2980B9', label='Negative')
    ]
    plt.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()