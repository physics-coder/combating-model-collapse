import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file into a DataFrame
df = pd.read_csv("perplexity_scores.csv")

# Create a mapping for directory renaming with abbreviated labels
rename_mapping = {
    "fast_webtext_control": "WebText Control",
    "fast_synth_control": "Synthetic Control",
    "0000": "0% R, 0% F",
    "0500": "5% R, 0% F",
    "1000": "10% R, 0% F",
    "1500": "15% R, 0% F",
    "2000": "20% R, 0% F",
    "2500": "25% R, 0% F",
    "50100": "25% R, 100% F"
}

# Apply the renaming to the DataFrame
df['Model'] = df['Model'].map(rename_mapping)

# Drop any rows where the model mapping failed (e.g., unmatched values)
df = df.dropna(subset=['Model'])

# Order the models as specified
ordered_models = [
    "WebText Control",
    "Synthetic Control",
    "0% R, 0% F",
    "5% R, 0% F",
    "10% R, 0% F",
    "15% R, 0% F",
    "20% R, 0% F",
    "25% R, 0% F",
    "25% R, 100% F"
]

# Sort the DataFrame based on the ordered models
df['Model'] = pd.Categorical(df['Model'], categories=ordered_models, ordered=True)
df = df.sort_values('Model')

# Set up the plot style
sns.set_theme(style="whitegrid")
sns.set_palette("deep")

# Function to plot perplexity for all models across epochs for a specific evaluation type
def plot_perplexity_comparison(df, eval_type):
    # Melt the DataFrame to long format for easier plotting
    id_vars = ['Model']
    value_vars = [f'epoch_{i}_eval_{eval_type}' for i in range(1, 4)]
    df_melted = df.melt(id_vars=id_vars, value_vars=value_vars, var_name='Epoch', value_name='Perplexity')

    # Create a mapping for clearer epoch names
    epoch_mapping = {
        f'epoch_1_eval_{eval_type}': 'Epoch 1',
        f'epoch_2_eval_{eval_type}': 'Epoch 2',
        f'epoch_3_eval_{eval_type}': 'Epoch 3'
    }

    # Apply the mapping to the Epoch column
    df_melted['Epoch'] = df_melted['Epoch'].map(epoch_mapping)

    # Create a new figure with A4 landscape size (11.69 x 8.1 inches)
    fig, ax = plt.subplots(figsize=(11.69, 8.1))

    # Plot lines for each epoch
    sns.lineplot(data=df_melted, x='Model', y='Perplexity', hue='Epoch', marker='o', ax=ax,
                 linewidth=3, markersize=10)

    # Set titles and labels with increased font sizes
    if eval_type == "synth":
        title = f'Perplexity Across Epochs for Different Models\n(Synthetic Evaluation)'
    else:
        title = f'Perplexity Across Epochs for Different Models\n({eval_type.capitalize()} Evaluation)'

    ax.set_title(title, fontsize=18, fontweight='bold', pad=30)
    ax.set_ylabel('Perplexity', fontsize=16, fontweight='bold', labelpad=15)
    ax.set_xlabel('Data Augmentation Configuration', fontsize=16, fontweight='bold', labelpad=15)

    # Adjust the x-axis labels
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.yticks(fontsize=14)

    # Increase tick label size and add padding
    ax.tick_params(axis='both', which='major', labelsize=14, pad=8)

    # Remove top and right spines
    sns.despine()

    # Make grid lines lighter
    ax.grid(True, linestyle='--', alpha=0.7)

    # Adjust layout to make room for legend and x-axis labels
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25, top=0.9)

    # Move the legend below the plot and adjust its position
    ax.legend(title='Epoch', title_fontsize='16', fontsize='14', loc='upper center',
              bbox_to_anchor=(0.5, -0.35), ncol=3)

    # Save the figure
    filename = f'perplexity_comparison_{eval_type}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')

    # Close the figure to avoid memory issues
    plt.close()

    print(f"Plot saved for {eval_type} evaluation")

# Call the plotting function for both webtext and synth evaluations
plot_perplexity_comparison(df, "webtext")
plot_perplexity_comparison(df, "synth")

print("Perplexity plots have been saved.")
