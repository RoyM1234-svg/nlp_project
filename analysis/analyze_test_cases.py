import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from models.verifier_model import VerifierModel
VERIFIER_QUESTION = "Who is the guilty suspect?"

def main():
    df = pd.read_csv("data/test_cases.csv")    
    verifier_model = VerifierModel("musr/verifier_model")
    
    for i, row in df.iterrows():
        index = row['Index']
        story = row['Story']
        suspects = row['Suspects']
        cot = row['Cot']

        text = (
            story
            + "\nSuspects: " + suspects
            + "\nQuestion: " + VERIFIER_QUESTION
            + "\nChain of Thought: " + cot
        )
        
        probs_correct = verifier_model.predict_prob_correct([text])
        prob = probs_correct[0] 
        
        print(f"Test case {index}: Probability correct = {prob:.4f}")
    

if __name__ == "__main__":
    main()