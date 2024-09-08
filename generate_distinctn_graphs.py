import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file into a DataFrame
df = pd.read_csv("gathered_diversity.csv")

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
    "5010": "25% R, 100% F"
}

# Apply the renaming to the DataFrame
df['Directory'] = df['Directory'].map(rename_mapping)

# Drop any rows where the directory mapping failed (e.g., unmatched values)
df = df.dropna(subset=['Directory'])

# Order the directories as specified
ordered_directories = [
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

# Sort the DataFrame based on the ordered directories
df['Directory'] = pd.Categorical(df['Directory'], categories=ordered_directories, ordered=True)
df = df.sort_values('Directory')

# Set up the plot style
sns.set_theme(style="whitegrid")
sns.set_palette("deep")

# Function to plot distinct-n metrics
def plot_distinct_metrics(experiment_data, experiment):
    # Create a new figure with A4 landscape size (11.69 x 8.27 inches)
    fig, ax = plt.subplots(figsize=(11.69, 8.27))

    # Plot distinct-1, distinct-2, distinct-3 using line plots with markers
    for metric in ['Distinct-1', 'Distinct-2', 'Distinct-3']:
        sns.lineplot(data=experiment_data, x='Directory', y=metric, marker='o', label=metric, ax=ax,
                     linewidth=3, markersize=10)

    # Improve the title
    if experiment == "nucleus_sampling":
        title = "Lexical Diversity Metrics Across Different Augmentation Strategies\nUsing Nucleus Sampling"
    elif experiment == "top_k":
        title = "Lexical Diversity Metrics Across Different Augmentation Strategies\nUsing Top-K Sampling"
    else:
        title = f"Lexical Diversity Metrics for {experiment.replace('_', ' ').title()} Experiment"

    ax.set_title(title, fontsize=18, fontweight='bold', pad=40)

    # Improve axis labels
    ax.set_ylabel('Distinct-n Score', fontsize=16, fontweight='bold', labelpad=15)
    ax.set_xlabel('Data Augmentation Configuration', fontsize=16, fontweight='bold', labelpad=15)

    # Adjust the x-axis labels
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.yticks(fontsize=14)

    # Adjust y-axis to include all data points
    y_min = experiment_data[['Distinct-1', 'Distinct-2', 'Distinct-3']].min().min() * 0.99
    y_max = experiment_data[['Distinct-1', 'Distinct-2', 'Distinct-3']].max().max() * 1.01
    ax.set_ylim(y_min, y_max)

    # Increase tick label size and add padding
    ax.tick_params(axis='both', which='major', labelsize=14, pad=8)

    # Remove top and right spines
    sns.despine()

    # Make grid lines lighter
    ax.grid(True, linestyle='--', alpha=0.7)

    # Adjust layout to make room for legend and x-axis labels
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)

    # Move the legend below the plot and adjust its position
    ax.legend(title='Lexical Diversity Metric', title_fontsize='16', fontsize='14', loc='upper center',
              bbox_to_anchor=(0.5, -0.35), ncol=3)

    # Save the figure with the original filename format
    filename = f'{experiment}_distinct_metrics.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')

    # Close the figure to avoid memory issues
    plt.close()

    print(f"Plot saved for {experiment}")

# Loop through each experiment to create and save a plot
for experiment in df['Experiment'].unique():
    # Filter the data for the current experiment
    experiment_data = df[df['Experiment'] == experiment]

    # Create the plot for this experiment
    plot_distinct_metrics(experiment_data, experiment)

print("All distinct-n metric plots have been saved.")
