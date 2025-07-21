import json, pandas as pd, pathlib
import os
from typing import Any, Dict, List, Tuple


def parse_chain_of_thought(tree_structure):
    def extract_reasoning_steps(node, depth=0):
        steps = []
        
        if isinstance(node, dict) and 'value' in node:
            step_text = node['value'].strip()
            if step_text:
                indent = "  " * depth
                steps.append(f"{indent}• {step_text}")
                
            if 'children' in node and node['children']:
                for child in node['children']:
                    steps.extend(extract_reasoning_steps(child, depth + 1))
        
        return steps
    
    if not tree_structure or not isinstance(tree_structure, dict):
        return "No reasoning chain available."
    
    # Try to get the root structure
    root_nodes = tree_structure.get('root_structure', tree_structure.get('nodes', []))
    
    if not root_nodes:
        return "No reasoning chain available."
    
    all_steps = []
    for root_node in root_nodes:
        all_steps.extend(extract_reasoning_steps(root_node))
    
    return "\n".join(all_steps) if all_steps else "No reasoning chain available."

def _dfs_collect(node: Dict[str, Any], steps: List[str]) -> None:
    if not isinstance(node, dict):
        return
    text = node.get("value", "").strip()
    if text:
        steps.append(text)
    for child in node.get("children", []):
        _dfs_collect(child, steps)

def parse_chain_of_thought_flat(tree: Dict[str, Any],
                                as_string: bool = True,
                                joiner: str = " ") -> Tuple[str, List[str]]:
    if not tree or not isinstance(tree, dict):
        return "" if as_string else []
    
    roots = tree.get("root_structure") or tree.get("nodes") or [tree]
    
    steps: List[str] = []
    for root in roots:
        _dfs_collect(root, steps)
    
    return (joiner.join(steps) if as_string else steps)

def build_dataset():
    
    script_dir = pathlib.Path(__file__).parent
    fpath = script_dir / "data" / "MUSR_json.json"

    with fpath.open(encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for story in data:
        mystery_txt = story["context"]

        for q_idx, question in enumerate(story["questions"]):
            suspects = question["choices"]
            murderer_idx = question["answer"]
            question_text = question.get("question", f"question_{q_idx}")

            murderer_name = suspects[murderer_idx]

            for i, suspect_name in enumerate(suspects):
                flat_cot = parse_chain_of_thought_flat(question["intermediate_trees"][i])

                row = {
                    "story":            mystery_txt,
                    "question":         question_text,
                    "suspects":         suspects,
                    "suspect_examined": suspect_name,
                    "flat_chain":       flat_cot,
                    "actual_murderer":  murderer_name,
                    "label":            int(i == murderer_idx)
                }

                rows.append(row)

    df = pd.DataFrame(rows)

    df.to_csv(script_dir / "verifier_training_data" / "MUSR_data.csv", index=False)

def analyze_musr_data():
    """
    for internal use run this function to get a sense of the data
    """
    script_dir = pathlib.Path(__file__).parent
    csv_path = script_dir / "verifier_training_data" / "MUSR_data.csv"
    
    if not csv_path.exists():
        print(f"Error: MUSR_data.csv not found at {csv_path}")
        print("Please run build_dataset() first to generate the CSV file.")
        return
    
    print("=" * 60)
    print("MUSR DATASET ANALYSIS")
    print("=" * 60)
    
    df = pd.read_csv(csv_path)
    
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"   Total rows: {len(df):,}")
    print(f"   Total columns: {len(df.columns)}")
    print(f"   Columns: {', '.join(df.columns)}")
    
    print(f"\n🎯 LABEL DISTRIBUTION:")
    label_counts = df['label'].value_counts()
    positive_examples = label_counts.get(1, 0)
    negative_examples = label_counts.get(0, 0)
    print(f"   Positive examples (murderer): {positive_examples:,}")
    print(f"   Negative examples (innocent): {negative_examples:,}")
    
    
    print(f"\n📝 TEXT LENGTH ANALYSIS:")
    df['story_length'] = df['story'].str.len()
    df['chain_length'] = df['flat_chain'].str.len()
    
    print(f"   Story length - Mean: {df['story_length'].mean():.0f} chars")
    print(f"   Story length - Median: {df['story_length'].median():.0f} chars")
    print(f"   Story length - Min/Max: {df['story_length'].min()}/{df['story_length'].max()}")
    
    print(f"   Chain length - Mean: {df['chain_length'].mean():.0f} chars")
    print(f"   Chain length - Median: {df['chain_length'].median():.0f} chars")
    print(f"   Chain length - Min/Max: {df['chain_length'].min()}/{df['chain_length'].max()}")
    
    print(f"\n❓ MISSING DATA:")
    missing_data = df.isnull().sum()
    if missing_data.sum() == 0:
        print("   No missing data found! ✅")
    else:
        for col, missing_count in missing_data.items():
            if missing_count > 0:
                print(f"   {col}: {missing_count:,} missing values ({missing_count/len(df)*100:.2f}%)")
    
    print(f"\n🔍 SAMPLE DATA:")
    print(f"   First positive example (murderer):")
    positive_sample = df[df['label'] == 1].iloc[0] if len(df[df['label'] == 1]) > 0 else None
    if positive_sample is not None:
        print(f"     Suspect: {positive_sample['suspect_examined']}")
        print(f"     Actual murderer: {positive_sample['actual_murderer']}")
        print(f"     Chain length: {len(positive_sample['flat_chain'])} chars")
    
    print(f"   First negative example (innocent):")
    negative_sample = df[df['label'] == 0].iloc[0] if len(df[df['label'] == 0]) > 0 else None
    if negative_sample is not None:
        print(f"     Suspect: {negative_sample['suspect_examined']}")
        print(f"     Actual murderer: {negative_sample['actual_murderer']}")
        print(f"     Chain length: {len(negative_sample['flat_chain'])} chars")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


def main():
    print("Building MUSR dataset...")
    build_dataset()
    print("Dataset built successfully!")
    
    print("\nRunning data analysis...")
    analyze_musr_data()

if __name__ == "__main__":
    main()
