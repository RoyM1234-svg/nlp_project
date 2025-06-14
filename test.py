from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.pipelines import pipeline
import torch
import argparse

def load_saved_model(model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        # torch_dtype=torch.float16,
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )
    
    return pipe

def run_inference(pipe, prompt, max_new_tokens=1000, temperature=0.7):
    """
    Run inference with the loaded model.
    
    Args:
        pipe: The loaded pipeline
        prompt: The input text to generate from
        max_new_tokens: Maximum number of tokens to generate
        temperature: Controls randomness (higher = more random)
    
    Returns:
        Generated text
    """
    output = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature
    )[0]["generated_text"]
    
    return output

# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="saved_llama_model")
    args = parser.parse_args()
    # Load the saved model
    model_path = args.model_path
    pipe = load_saved_model(model_path)
    
    # Example prompt
    prompt = """<s>[INST] You are a detective solving a mystery. Read the following case carefully and identify the Flying Bandit from the four suspects. Pay close attention to the details and use logical reasoning to solve this mystery.

    Mystery: Sergeant Saunders of Scotland Yard hung up the telephone and turned to his superior who was studying a map of the world tacked to one wall of his office. "No luck finding the money yet, sir," Saunders reported. "A knapsack full of two-hundred-thousand American dollars is a tricky thing to track down in all that expanse."

    Chief Inspector Langford turned away from the map, spun the large globe that sat on a corner of his desk. "Quite," he agreed. "Perhaps the Flying Bandit himself can help us out with the search. Once we capture him." He placed his cold pipe between his teeth and took a couple of smokeless puffs. The Chief Inspector was a non-smoker, but he felt that his position required a pipe for dramatic effect. Thus, he always carried one for show but would never light it.

    Saunders nodded. "Right, we know he took off from a Long Island airfield after he robbed the Chase Manhattan Bank in New York City, and that he landed on the outskirts of Liverpool fifteen hours later – witnesses identified the plane taking off and landing at those two points. Given the time interval, and an airspeed of 200 miles-per-hour, we know he took the most direct route to get to England, approximately 3,000 miles as the crow, or bandit, flies. It was a one-man flight his fellow countryman, Charles Lindbergh, would've been proud of."

    "And almost exactly thirty years to the day that Lindy landed in Paris," Langford mused, tracing Lindbergh's solo route with his finger on the globe.

    "We also know," Saunders continued, "that the Flying Bandit lost the knapsack full of money approximately five hours into his getaway flight. We surmise that from the fact that the American police heard a scream of anguish as he was bragging to them on the radio about his 'perfect' crime and getaway. They believe he was dangling the loot out the window of the plane in a fit of showmanship and dropped it."

    "Yes, quite. Bad luck on his part," Langford murmured. The tall, gaunt man with the pork chop sideburns jabbed a spot on the globe with a bony finger, indicating the probable location of the money drop.

    "But he continued on-course all the way to Liverpool, probably figuring the Americans would have their own planes out looking for him," Saunders went on. "But now that things have cooled down a bit, he'll be anxious to retrace his route and find that knapsack of cash he let slip out of his fingers."

    "Commendable summary of the known facts, Saunders," Langford commented dryly. "With all British exit points on high alert, we're certain to-"

    The phone rang again. Sergeant Saunders scooped it up, listened for a moment, said, "Right-o!", and then pronged the receiver. "They've rounded up four men trying to leave the country who match the Flying Bandit's general description, sir. They're in the holding cells right now, waiting to be questioned."

    Chief Inspector Langford smiled, tapping his teeth with his black pipe stem. "Good. We're fortunate the man's aircraft broke both wheels upon landing, or he'd certainly have used that bird to fly the coop undetected from our shores. His bad luck, again."

    The man in Holding Cell 1 at Scotland Yard was David Loftkiss, a jeweler by profession, so he claimed, in the United Kingdom on business.

    "Mr. Loftkiss was stopped by the Southampton constabulary trying to board the SS United States bound for New York City," Saunders summarized the report he'd received.

    "I have urgent business in New York!" Loftkiss exclaimed, rising out of his chair. "Why are you holding me up?"

    Langford calmly compared the man's face to the police sketch the Yard had received of the Flying Bandit, based on the bank president's description. "Indeed. Urgent business, you say? Why not take a plane, then, Mr. Loftkiss? Why take an ocean liner?"

    The man sat back down, his face pale. "I-I'm afraid of flying. I don't trust those propellers to keep spinning – so high up in the air. So I take boats whenever I go overseas. And the SS United States is the fastest ocean liner on the waters. No stops."

    "Hmmm," Langford commented.

    "This is 1957, Mr. Loftkiss," Saunders stated. "Air safety has improved considerably since your country's barnstorming days."

    "Maybe. But I don't take any chances. So, can I-"

    The two policemen moved on to Holding Cell 2. Sitting on a wobbly wooden chair at a worn wooden table was Cliff Snelling, artist, so he claimed. The resemblance between him and the Flying Bandit, as with Loftkiss, was strong.

    Langford contemplatively tapped his pipe, and said, "Paint me a picture, Mr. Snelling. Why were you attempting to board a ferry at Holyhead, bound for Dublin, Ireland? Perhaps trying to retrace your original flight path after the robbery?"

    "What? What 'flight path'!" Snelling exploded, exhibiting a highly volatile artistic temperament, if nothing else.

    "I've been commissioned to paint some horses at Lord Harding's Irish estate," the agitated man added. "That's why I was taking the ferry to Dublin. Please, I-"

    "And to think I always thought horses came with their own natural coloring," Langford deadpanned to his assistant.

    The two policemen moved on to Holding Cell 3. Wherein, a Mr. Tom Jenks presided over a stony-faced silence, his arms folded across his chest.

    "You were picked up at Heathrow Airport," the Chief Inspector stated, drawing upon his cold pipe as he did so the facts. "What was your exact intended destination, sir?"

    "Blow it out your briar, bobby!" Jenks retorted. "I'm suing you blokes for false arrest and imprisonment. This is how you treat American tourists – after we won the war for you!"

    "Yes, well, we're all very grateful for that, I'm sure," Langford puffed.

    "The airline says he was bound for Gander, Newfoundland, sir," Sergeant Saunders informed his superior.

    "More tourism," Jenks said. "I thought I'd see some of Canada on my way back to New York, that's all. Now, you flatfoots had better-"

    The two policemen exchanged glances, and then moved on to Holding Cell 4. Occupying this barren, yellow-walled room was Clem Duster.

    "As I told the coppers who picked me up at the airfield, I was just renting a plane to do some pleasure-flying while I was in Manchester. I was a pilot in the Army Air Force during the war, see, and-"

    "Those 'coppers'," Langford interrupted, "found a flight plan to Iceland in that rented plane of yours, Mr. Duster. Iceland seems a tad far for a 'pleasure flight', doesn't it?"

    "Okay, okay!" the man conceded. "Don't tell my wife, but I met this dame, Miss Iceland, at a beauty pageant in New York a few weeks ago, and she told me about all these great hot springs they have in her home country. So, with my skin condition and all, I figured …"

    The two policemen left the cell and returned to Langford's office. The Chief Inspector's pipe was cold, but not the trail to nabbing the Flying Bandit. "Well, I'd say we've got our chap, wouldn't you, Saunders?"

    The Sergeant nodded, his ginger mustache bristling upwards into a smile. "Shall I go get him out of the holding cell and bring him in here to be formally charged?"

    Langford knocked his pipe against the globe on his desk over the general area where the stolen money had been dropped. "Quite. And by the least circuitous route possible, if you don't mind," he added, his sky-blue eyes twinkling.

    Please provide your answer in the following format:
    Reasoning: [your step-by-step reasoning for why this is the correct answer]
    Answer: [one of the four suspects: David Loftkiss, Cliff Snelling, Tom Jenks, or Clem Duster] [/INST]"""
        
    # Run inference
    result = run_inference(pipe, prompt)
    print(f"Output: {result}")
