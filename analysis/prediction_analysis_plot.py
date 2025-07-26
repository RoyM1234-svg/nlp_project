import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def create_prediction_analysis_plot(csv_file_path, verifier_csv_path=None, output_path=None):
    """
    Create separate bar plots showing the number of unknown, correct, and incorrect predictions from a CSV file,
    and optionally mean probabilities if verifier CSV is provided.
    
    Args:
        csv_file_path (str): Path to the CSV file containing predictions
        verifier_csv_path (str, optional): Path to the verifier CSV file containing probabilities
        output_path (str, optional): Base path to save the plots. If None, shows the plots.
    """
    
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: File '{csv_file_path}' not found.")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    required_columns = ['predictions', 'true_labels']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        return
    
    unknown_count = (df['predictions'] == 'Unknown').sum()
    
    non_unknown_mask = df['predictions'] != 'Unknown'
    correct_count = ((df['predictions'] == df['true_labels']) & non_unknown_mask).sum()
    incorrect_count = ((df['predictions'] != df['true_labels']) & non_unknown_mask).sum()
    
    categories = ['Unknown', 'Correct', 'Incorrect']
    counts = [unknown_count, correct_count, incorrect_count]
    colors = ['#FFA726', '#4CAF50', '#FF6B6B']  
    
    plt.figure(1, figsize=(12, 7))
    bars1 = plt.bar(categories, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    plt.title('Prediction Counts: Unknown vs Correct vs Incorrect', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Prediction Type', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Predictions', fontsize=14, fontweight='bold')
    
    for bar, count in zip(bars1, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(counts) * 0.01,
                f'{count}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    total_predictions = len(df)
    unknown_pct = (unknown_count / total_predictions) * 100
    correct_pct = (correct_count / total_predictions) * 100
    incorrect_pct = (incorrect_count / total_predictions) * 100
    
    for i, (count, pct) in enumerate(zip(counts, [unknown_pct, correct_pct, incorrect_pct])):
        if count > total_predictions * 0.05:  
            plt.text(i, count/2, f'{pct:.1f}%', ha='center', va='center', 
                     fontsize=11, fontweight='bold', color='white')
    
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    accuracy_pct = (correct_count / (correct_count + incorrect_count)) * 100 if (correct_count + incorrect_count) > 0 else 0
    count_summary = (f'Total: {total_predictions} | Unknown: {unknown_count} ({unknown_pct:.1f}%) | '
                    f'Correct: {correct_count} ({correct_pct:.1f}%) | Incorrect: {incorrect_count} ({incorrect_pct:.1f}%) | '
                    f'Accuracy (excl. Unknown): {accuracy_pct:.1f}%')
    plt.figtext(0.5, 0.02, count_summary, ha='center', fontsize=10, style='italic')
    
    if output_path:
        count_output_path = output_path.replace('.png', '_counts.png') if output_path.endswith('.png') else f"{output_path}_counts.png"
        plt.savefig(count_output_path, dpi=300, bbox_inches='tight')
        print(f"Count plot saved to: {count_output_path}")
    else:
        plt.show()
    
    has_verifier = verifier_csv_path is not None
    mean_prob_unknown = mean_prob_correct = mean_prob_incorrect = 0
    
    if has_verifier:
        try:
            verifier_df = pd.read_csv(verifier_csv_path)
            
            if 'case_names' not in verifier_df.columns or 'probs_correct' not in verifier_df.columns:
                print(f"Warning: Verifier CSV missing required columns. Skipping probability analysis.")
                has_verifier = False
            else:
                merged_df = pd.merge(df, verifier_df[['case_names', 'probs_correct']], 
                                   on='case_names', how='inner', suffixes=('', '_verifier'))
                
                if len(merged_df) == 0:
                    print("Warning: No matching case_names found between the two CSV files.")
                    has_verifier = False
                else:
                    unknown_mask = merged_df['predictions'] == 'Unknown'
                    correct_mask = ((merged_df['predictions'] == merged_df['true_labels']) & 
                                  (merged_df['predictions'] != 'Unknown'))
                    incorrect_mask = ((merged_df['predictions'] != merged_df['true_labels']) & 
                                    (merged_df['predictions'] != 'Unknown'))
                    
                    mean_prob_unknown = merged_df.loc[unknown_mask, 'probs_correct'].mean() if unknown_mask.sum() > 0 else 0
                    mean_prob_correct = merged_df.loc[correct_mask, 'probs_correct'].mean() if correct_mask.sum() > 0 else 0
                    mean_prob_incorrect = merged_df.loc[incorrect_mask, 'probs_correct'].mean() if incorrect_mask.sum() > 0 else 0
                    
                    mean_probs = [mean_prob_unknown, mean_prob_correct, mean_prob_incorrect]
                    
                    plt.figure(2, figsize=(12, 7))
                    bars2 = plt.bar(categories, mean_probs, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
                    
                    plt.title('Mean Verifier Probabilities by Prediction Type', fontsize=16, fontweight='bold', pad=20)
                    plt.xlabel('Prediction Type', fontsize=14, fontweight='bold')
                    plt.ylabel('Mean Probability', fontsize=14, fontweight='bold')
                    plt.ylim(0, 1)
                    
                    for bar, prob in zip(bars2, mean_probs):
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                f'{prob:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
                    
                    plt.grid(axis='y', alpha=0.3, linestyle='--')
                    plt.tight_layout()
                    
                    prob_summary = (f'Mean Probabilities - Unknown: {mean_prob_unknown:.3f} | '
                                   f'Correct: {mean_prob_correct:.3f} | Incorrect: {mean_prob_incorrect:.3f}')
                    plt.figtext(0.5, 0.02, prob_summary, ha='center', fontsize=10, style='italic')
                    
                    if output_path:
                        prob_output_path = output_path.replace('.png', '_probabilities.png') if output_path.endswith('.png') else f"{output_path}_probabilities.png"
                        plt.savefig(prob_output_path, dpi=300, bbox_inches='tight')
                        print(f"Probability plot saved to: {prob_output_path}")
                    else:
                        plt.show()
                    
        except FileNotFoundError:
            print(f"Warning: Verifier file '{verifier_csv_path}' not found. Skipping probability analysis.")
            has_verifier = False
        except Exception as e:
            print(f"Warning: Error reading verifier CSV file: {e}. Skipping probability analysis.")
            has_verifier = False
    
    # Create the third figure for accuracy progression (k=1 to k=3)
    plt.figure(3, figsize=(12, 7))
    
    # Data points for k=1 and k=3
    k_values = [1, 3]
    accuracy_values = [27.2, 30.4]
    
    # Create smooth line with single color
    # Plot the smooth line from k=1 to k=3
    plt.plot(k_values, accuracy_values, '-', color='#4CAF50', linewidth=4, label='Accuracy progression')
    
    # Plot the data points
    plt.plot([k_values[0]], [accuracy_values[0]], 'o', color='#4CAF50', markersize=12)
    plt.plot([k_values[1]], [accuracy_values[1]], 'o', color='#4CAF50', markersize=12)
    
    # Customize the plot
    plt.title('Accuracy Improvement from k=1 to k=3', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Self Consistency Variable k', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    plt.xlim(0.5, 3.5)
    plt.ylim(25, 32)
    
    # Add value labels
    plt.text(k_values[0], accuracy_values[0] + 0.3, f'{accuracy_values[0]:.1f}%', 
             ha='center', va='bottom', fontsize=12, fontweight='bold', color='#4CAF50')
    plt.text(k_values[1], accuracy_values[1] + 0.3, f'{accuracy_values[1]:.1f}%', 
             ha='center', va='bottom', fontsize=12, fontweight='bold', color='#4CAF50')
    
    # Customize grid and layout
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.xticks(k_values)
    plt.tight_layout()
    
    # Add summary text
    improvement = accuracy_values[1] - accuracy_values[0]
    accuracy_summary = (f'k=1 Accuracy: {accuracy_values[0]:.1f}% | k=3 Accuracy: {accuracy_values[1]:.1f}% | '
                       f'Improvement: +{improvement:.1f} percentage points')
    plt.figtext(0.5, 0.02, accuracy_summary, ha='center', fontsize=10, style='italic')
    
    # Save or show the accuracy plot
    if output_path:
        accuracy_output_path = output_path.replace('.png', '_accuracy.png') if output_path.endswith('.png') else f"{output_path}_accuracy.png"
        plt.savefig(accuracy_output_path, dpi=300, bbox_inches='tight')
        print(f"Accuracy plot saved to: {accuracy_output_path}")
    else:
        plt.show()
    
    print(f"\nPrediction Analysis Summary:")
    print(f"Total predictions: {total_predictions}")
    print(f"Unknown predictions: {unknown_count} ({unknown_pct:.1f}%)")
    print(f"Correct predictions: {correct_count} ({correct_pct:.1f}%)")
    print(f"Incorrect predictions: {incorrect_count} ({incorrect_pct:.1f}%)")
    if (correct_count + incorrect_count) > 0:
        print(f"Accuracy (excluding Unknown): {accuracy_pct:.1f}%")
    
    if has_verifier:
        print(f"\nMean Verifier Probabilities:")
        print(f"Unknown predictions: {mean_prob_unknown:.3f}")
        print(f"Correct predictions: {mean_prob_correct:.3f}")
        print(f"Incorrect predictions: {mean_prob_incorrect:.3f}")
    
    print(f"\nAccuracy Progression:")
    print(f"k=1 Accuracy: {accuracy_values[0]:.1f}%")
    print(f"k=3 Accuracy: {accuracy_values[1]:.1f}%")
    print(f"Improvement: +{improvement:.1f} percentage points")
    
    result = {
        'total': total_predictions,
        'unknown': unknown_count,
        'correct': correct_count,
        'incorrect': incorrect_count,
        'unknown_pct': unknown_pct,
        'correct_pct': correct_pct,
        'incorrect_pct': incorrect_pct,
        'accuracy_excl_unknown': accuracy_pct,
        'k1_accuracy': accuracy_values[0],
        'k3_accuracy': accuracy_values[1],
        'improvement': improvement
    }
    
    if has_verifier:
        result.update({
            'mean_prob_unknown': mean_prob_unknown,
            'mean_prob_correct': mean_prob_correct,
            'mean_prob_incorrect': mean_prob_incorrect
        })
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Create separate bar plots showing prediction counts and optionally mean probabilities from CSV files')
    parser.add_argument('csv_file', help='Path to the CSV file containing predictions')
    parser.add_argument('-v', '--verifier', help='Path to the verifier CSV file containing probabilities (optional)')
    parser.add_argument('-o', '--output', help='Base output path for the plots (will add _counts.png and _probabilities.png suffixes)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_file):
        print(f"Error: File '{args.csv_file}' does not exist.")
        return
    
    if args.verifier and not os.path.exists(args.verifier):
        print(f"Error: Verifier file '{args.verifier}' does not exist.")
        return
    
    create_prediction_analysis_plot(args.csv_file, args.verifier, args.output)

if __name__ == "__main__":
    main() 