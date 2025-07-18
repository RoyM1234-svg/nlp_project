#!/usr/bin/env python3
"""
Analysis script for DeepSeek model results CSV file
Provides comprehensive analysis of model predictions and chain-of-thought reasoning.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import argparse
import os

def load_data(filepath):
    """Load the CSV data and handle any parsing issues."""
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded {len(df)} rows from {filepath}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def analyze_predictions(df):
    """Analyze the predictions column."""
    print("\n" + "="*50)
    print("PREDICTIONS ANALYSIS")
    print("="*50)
    
    # Count Unknown predictions
    unknown_count = (df['predictions'] == 'Unknown').sum()
    total_predictions = len(df)
    unknown_percentage = (unknown_count / total_predictions) * 100
    
    print(f"Total predictions: {total_predictions}")
    print(f"Unknown predictions: {unknown_count}")
    print(f"Unknown percentage: {unknown_percentage:.2f}%")
    print(f"Valid predictions: {total_predictions - unknown_count}")
    
    # Most common predictions
    print(f"\nTop 10 most common predictions:")
    prediction_counts = df['predictions'].value_counts().head(10)
    for pred, count in prediction_counts.items():
        print(f"  {pred}: {count} ({count/total_predictions*100:.1f}%)")
    
    return {
        'total': total_predictions,
        'unknown_count': unknown_count,
        'unknown_percentage': unknown_percentage,
        'valid_predictions': total_predictions - unknown_count
    }

def analyze_accuracy(df):
    """Calculate accuracy metrics."""
    print("\n" + "="*50)
    print("ACCURACY ANALYSIS")
    print("="*50)
    
    # Overall accuracy
    correct_predictions = (df['predictions'] == df['true_labels']).sum()
    total_predictions = len(df)
    overall_accuracy = (correct_predictions / total_predictions) * 100
    
    print(f"Overall accuracy: {overall_accuracy:.2f}% ({correct_predictions}/{total_predictions})")
    
    # Accuracy excluding Unknown predictions
    valid_mask = df['predictions'] != 'Unknown'
    valid_df = df[valid_mask]
    
    if len(valid_df) > 0:
        valid_correct = (valid_df['predictions'] == valid_df['true_labels']).sum()
        valid_total = len(valid_df)
        valid_accuracy = (valid_correct / valid_total) * 100
        print(f"Accuracy (excluding Unknown): {valid_accuracy:.2f}% ({valid_correct}/{valid_total})")
    else:
        valid_accuracy = 0
        print("No valid predictions to calculate accuracy")
    
    # Accuracy when prediction is Unknown
    unknown_mask = df['predictions'] == 'Unknown'
    unknown_df = df[unknown_mask]
    
    if len(unknown_df) > 0:
        # This would be 0% since Unknown != actual labels
        unknown_accuracy = 0
        print(f"Accuracy for Unknown predictions: {unknown_accuracy:.2f}%")
    
    return {
        'overall_accuracy': overall_accuracy,
        'valid_accuracy': valid_accuracy,
        'correct_predictions': correct_predictions,
        'valid_correct': valid_correct if len(valid_df) > 0 else 0
    }

def analyze_cots(df):
    """Analyze chain-of-thought (generated_cots) lengths and content."""
    print("\n" + "="*50)
    print("CHAIN-OF-THOUGHT ANALYSIS")
    print("="*50)
    
    # Calculate lengths
    cot_lengths = df['generated_cots'].astype(str).str.len()
    
    print(f"Chain-of-thought length statistics:")
    print(f"  Minimum length: {cot_lengths.min()} characters")
    print(f"  Maximum length: {cot_lengths.max()} characters")
    print(f"  Mean length: {cot_lengths.mean():.2f} characters")
    print(f"  Median length: {cot_lengths.median():.2f} characters")
    print(f"  Standard deviation: {cot_lengths.std():.2f} characters")
    
    # Length distribution
    print(f"\nLength distribution:")
    print(f"  < 1000 chars: {(cot_lengths < 1000).sum()} ({(cot_lengths < 1000).sum()/len(df)*100:.1f}%)")
    print(f"  1000-2000 chars: {((cot_lengths >= 1000) & (cot_lengths < 2000)).sum()} ({((cot_lengths >= 1000) & (cot_lengths < 2000)).sum()/len(df)*100:.1f}%)")
    print(f"  2000-3000 chars: {((cot_lengths >= 2000) & (cot_lengths < 3000)).sum()} ({((cot_lengths >= 2000) & (cot_lengths < 3000)).sum()/len(df)*100:.1f}%)")
    print(f"  > 3000 chars: {(cot_lengths >= 3000).sum()} ({(cot_lengths >= 3000).sum()/len(df)*100:.1f}%)")
    
    # Analyze content patterns
    think_pattern_count = df['generated_cots'].astype(str).str.contains('<think>', case=False, na=False).sum()
    print(f"\nChains containing '<think>' pattern: {think_pattern_count} ({think_pattern_count/len(df)*100:.1f}%)")
    
    return {
        'min_length': cot_lengths.min(),
        'max_length': cot_lengths.max(),
        'mean_length': cot_lengths.mean(),
        'median_length': cot_lengths.median(),
        'std_length': cot_lengths.std(),
        'think_pattern_count': think_pattern_count
    }

def analyze_errors(df):
    """Analyze common error patterns and failure modes."""
    print("\n" + "="*50)
    print("ERROR ANALYSIS")
    print("="*50)
    
    # Cases where prediction is wrong
    wrong_mask = df['predictions'] != df['true_labels']
    wrong_df = df[wrong_mask]
    
    print(f"Total wrong predictions: {len(wrong_df)} ({len(wrong_df)/len(df)*100:.1f}%)")
    
    if len(wrong_df) > 0:
        # Analyze wrong predictions by type
        unknown_wrong = (wrong_df['predictions'] == 'Unknown').sum()
        other_wrong = len(wrong_df) - unknown_wrong
        
        print(f"  Wrong due to 'Unknown': {unknown_wrong} ({unknown_wrong/len(df)*100:.1f}%)")
        print(f"  Wrong with specific prediction: {other_wrong} ({other_wrong/len(df)*100:.1f}%)")
        
        # Show some examples of wrong predictions (excluding Unknown)
        specific_wrong = wrong_df[wrong_df['predictions'] != 'Unknown']
        if len(specific_wrong) > 0:
            print(f"\nExamples of specific wrong predictions (first 5):")
            for i, (_, row) in enumerate(specific_wrong.head().iterrows()):
                print(f"  {i+1}. Predicted: '{row['predictions']}', Actual: '{row['true_labels']}' (Index: {row['indices']})")

def create_visualizations(df, stats, model_name):
    """Create visualizations of the analysis (without predictions distribution plot)."""
    print("\n" + "="*50)
    print("CREATING VISUALIZATIONS")
    print("="*50)
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'{model_name} Results Analysis', fontsize=16, fontweight='bold')
    
    # 1. Chain-of-thought length distribution
    ax1 = axes[0]
    cot_lengths = df['generated_cots'].astype(str).str.len()
    ax1.hist(cot_lengths, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(cot_lengths.mean(), color='red', linestyle='--', label=f'Mean: {cot_lengths.mean():.0f}')
    ax1.axvline(cot_lengths.median(), color='orange', linestyle='--', label=f'Median: {cot_lengths.median():.0f}')
    ax1.set_xlabel('Chain-of-Thought Length (characters)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Chain-of-Thought Lengths')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Accuracy breakdown
    ax2 = axes[1]
    accuracy_data = {
        'Overall': stats['accuracy']['overall_accuracy'],
        'Valid Only\n(no Unknown)': stats['accuracy']['valid_accuracy']
    }
    bars = ax2.bar(accuracy_data.keys(), accuracy_data.values(), 
                   color=['lightcoral', 'lightgreen', 'lightgray'], alpha=0.8)
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy Breakdown')
    ax2.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Unknown vs Valid predictions
    ax3 = axes[2]
    unknown_valid_data = [stats['predictions']['unknown_count'], stats['predictions']['valid_predictions']]
    labels = ['Unknown', 'Valid Predictions']
    colors = ['lightcoral', 'lightgreen']
    bars = ax3.bar(labels, unknown_valid_data, color=colors, alpha=0.8)
    ax3.set_ylabel('Count')
    ax3.set_title('Unknown vs Valid Predictions')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height / 2,
                f'{int(height)}', ha='center', va='center', fontweight='bold', color='black')
    
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_filename = f'v2_plots/{model_name.lower()}_analysis.png'
    os.makedirs('v2_plots', exist_ok=True)
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Visualizations saved as '{output_filename}'")
    
    return fig

def generate_summary_report(stats):
    """Generate a summary report of all analyses."""
    print("\n" + "="*60)
    print("SUMMARY REPORT")
    print("="*60)
    
    print(f"📊 DATASET OVERVIEW:")
    print(f"   • Total samples: {stats['predictions']['total']}")
    print(f"   • Valid predictions: {stats['predictions']['valid_predictions']}")
    print(f"   • Unknown predictions: {stats['predictions']['unknown_count']} ({stats['predictions']['unknown_percentage']:.1f}%)")
    
    print(f"\n🎯 ACCURACY METRICS:")
    print(f"   • Overall accuracy: {stats['accuracy']['overall_accuracy']:.2f}%")
    print(f"   • Accuracy (excluding Unknown): {stats['accuracy']['valid_accuracy']:.2f}%")
    print(f"   • Correct predictions: {stats['accuracy']['correct_predictions']}/{stats['predictions']['total']}")
    
    print(f"\n📝 CHAIN-OF-THOUGHT ANALYSIS:")
    print(f"   • Min length: {stats['cots']['min_length']:,} characters")
    print(f"   • Max length: {stats['cots']['max_length']:,} characters")
    print(f"   • Average length: {stats['cots']['mean_length']:,.0f} characters")
    print(f"   • Median length: {stats['cots']['median_length']:,.0f} characters")
    print(f"   • Contains '<think>' pattern: {stats['cots']['think_pattern_count']} cases ({stats['cots']['think_pattern_count']/stats['predictions']['total']*100:.1f}%)")
    
    print(f"\n💡 KEY INSIGHTS:")
    if stats['predictions']['unknown_percentage'] > 20:
        print(f"   ⚠️  High rate of Unknown predictions ({stats['predictions']['unknown_percentage']:.1f}%) suggests model uncertainty")
    else:
        print(f"   ✅ Reasonable rate of Unknown predictions ({stats['predictions']['unknown_percentage']:.1f}%)")
    
    if stats['accuracy']['valid_accuracy'] > 70:
        print(f"   ✅ Good accuracy on valid predictions ({stats['accuracy']['valid_accuracy']:.1f}%)")
    elif stats['accuracy']['valid_accuracy'] > 50:
        print(f"   ⚠️  Moderate accuracy on valid predictions ({stats['accuracy']['valid_accuracy']:.1f}%)")
    else:
        print(f"   ❌ Low accuracy on valid predictions ({stats['accuracy']['valid_accuracy']:.1f}%)")
    
    cot_avg = stats['cots']['mean_length']
    if cot_avg > 2000:
        print(f"   📖 Detailed reasoning (avg {cot_avg:,.0f} chars) - model provides thorough analysis")
    elif cot_avg > 1000:
        print(f"   📄 Moderate reasoning length (avg {cot_avg:,.0f} chars)")
    else:
        print(f"   📝 Brief reasoning (avg {cot_avg:,.0f} chars) - may lack detail")

def main():
    """Main analysis function."""
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Analyze model results CSV file')
    parser.add_argument('csv_file', nargs='?', default='results_deepseek.csv',
                       help='Path to the CSV file (default: results_deepseek.csv)')
    parser.add_argument('--model-name', '-m', type=str, default=None,
                       help='Model name for titles and output files (auto-detected if not provided)')
    
    args = parser.parse_args()
    csv_file = args.csv_file
    
    # Extract model name from filename for titles
    if args.model_name:
        model_name = args.model_name
    else:
        # Auto-detect model name from filename
        filename = os.path.basename(csv_file)
        if 'llama' in filename.lower():
            model_name = 'LLaMA'
        elif 'deepseek' in filename.lower():
            model_name = 'DeepSeek'
        elif 'gemma' in filename.lower():
            model_name = 'Gemma'
        elif 'gpt' in filename.lower():
            model_name = 'GPT'
        elif 'claude' in filename.lower():
            model_name = 'Claude'
        else:
            # Extract from results_<model>.csv pattern
            base_name = filename.replace('.csv', '').replace('results_', '')
            model_name = base_name.replace('_', ' ').title() if base_name else 'Model'
    
    print(f"🔍 {model_name} Results Analysis")
    print("=" * 60)
    print(f"Analyzing: {csv_file}")
    
    # Load data
    df = load_data(csv_file)
    if df is None:
        return
    
    # Print basic info about the dataset
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Perform analyses
    prediction_stats = analyze_predictions(df)
    accuracy_stats = analyze_accuracy(df)
    cot_stats = analyze_cots(df)
    analyze_errors(df)
    
    # Combine all stats
    stats = {
        'predictions': prediction_stats,
        'accuracy': accuracy_stats,
        'cots': cot_stats
    }
    
    # Create visualizations
    try:
        create_visualizations(df, stats, model_name)
    except Exception as e:
        print(f"Error creating visualizations: {e}")
    
    # Generate summary report
    generate_summary_report(stats)
    
    print(f"\n✅ Analysis complete!")

if __name__ == "__main__":
    main() 