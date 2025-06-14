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
        torch_dtype=torch.float16,
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )
    
    return pipe


def create_basic_cot_prompt(suspects, mystery_text):
    """
    Basic Chain-of-Thought prompt that works for any mystery.
    """
    suspects_list = "\n".join([f"- {suspect}" for suspect in suspects])
    
    prompt = f"""<s>[INST] You are a detective solving a mystery. Your task is to identify the guilty suspect from the following suspects based on the evidence provided.

    SUSPECTS:
    {suspects_list}

    INSTRUCTIONS:
    1. Read the mystery carefully
    2. Identify key evidence and clues
    3. Analyze each suspect's alibi and behavior
    4. Use logical reasoning to eliminate innocent suspects
    5. Identify the guilty suspect based on evidence

    Think through this step-by-step:
    - First, what is the crime and what are the key facts?
    - Second, what evidence points to each suspect?
    - Third, which suspects can be eliminated and why?
    - Finally, who is guilty and what proves it?

    REQUIRED OUTPUT FORMAT:
    Reasoning: [Step-by-step analysis of each suspect and the evidence for/against them]
    Answer: [Name of the guilty suspect]

    MYSTERY:
    {mystery_text}
    [/INST]"""
    
    return prompt

def run_inference(pipe, prompt, max_new_tokens=100, temperature=0.7):
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
    print("starting inference...")

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
    
    mystery_text = """
    Lightning flashed through the castle window, before the sky returned to darkness and rain.

I trudged down the hallway, my candle flickering in invisible drafts from the stormy chill outside. Suddenly the candle lit up a face!

"Evening, squire," said Bertram, the care-taker.

"You startled me," I said, though I was glad he was there. I'd started to investigate the mystery of the scarecrow mask, and there were a lot of questions.

It was a small Scottish castle, miles away from the nearest town. Behind a large wooden door in the front, I discovered the whole first floor was a single, enormous room. Dark stone staircases led on either side to an open second floor, which was filled with shiny armor from the Middle Ages. It felt like the ghostly bodies of knights, preserved for centuries and now left on display. Above that was a dark top floor, which Bertram had converted into rooms for guests. But unfortunately, he'd only attracted three paying customers.

"And now I've lost me prime exhibit," he wailed.

Two of his guests emerged from their room - Mr. and Mrs. Winfrey. The elderly couple had heard our voices, and were wondering if Bertram was serving dinner. Mr. Winfrey was a large man who looked like he'd enjoyed many dinners. But Mrs. Winfrey just seemed unhappy to be so far from home, especially in a castle where it rained all day, complete with its own spooky mystery.

"I'll tell you how it began," Bertram said, as we gathered around a long wooden table downstairs. Bertram's son Chester had prepared a traditional dinner in the cook's quarters outside. As we munched on roasted chicken and potatoes, Bertram described a fierce battle that never was.

"The sheriff was searching for a dangerous outlaw," he said. "He'd watched the roads, and searched the fields, and determined that he must be hiding here, in this very castle. He sent for the king's soldiers, who responded with 100 men, each armed with swords and spears. That dangerous outlaw would not escape their justice.” Lightning flashed outside.

"They searched the first floor - but he was not here. They searched every suit of armor - but he was not there. And then they searched the rooms on the third floor. But there was no trace of him. From dawn they searched, and through dusk they searched, but the outlaw was never found.”

Mrs. Winfrey seemed confused, and she'd obviously never heard the legend before. "Where was he?" she asked.

"He was right here the whole time," Bertram warned. "He watched them come, and he watched them go. At any time, he could've reached out and touched them.”

"He stood out in the field, in front of them, in plain sight. But he'd dressed himself as a scarecrow. All through the day he stood, arms stretched out on either side, like an innocent straw man propped up to frighten away birds.”

We all laughed, and Bertram most of all. "It's what made this castle famous!" he chuckled, "And I've loved that story since I was a boy.”

"When I came here I vowed I'd preserve what became known as 'the Castle of the Scarecrow,' along with its heritage," he said seriously. "Amid all the armor upstairs, there's just one piece that the tourists will come to see - the mask of that outlaw scarecrow.

"But now it's gone!" Silence filled the room, as we heard the rain beat against the door.

"That's outrageous," said Mr. Winfrey. "We've traveled over 3,000 miles just to see that mask! My re-telling of that legend could make me a best-selling author.”

Mrs. Winfrey added proudly that her husband was writing an exciting book about the history of famous outlaws, then sighed that she was just along for the trip. I wondered if she was miserable enough to sabotage the castle's prime exhibit.

Bertram explained the crime. The scarecrow's mask stood on a pole at the center of the armor exhibit upstairs - as though surrounded by the armor of knights that it had fooled. Twenty minutes ago, shortly before opening the exhibit, Bertram entered the floor, only to see that pole standing conspicuously bare and empty.

"Maybe the scarecrow's ghost took it," said a sharp voice from the stairway.

A well-dressed man descended, the third guest at the rainy castle. "I'm proud of my family tree," said Charles Kincaid, defiantly. "And of all the generations that went before, there was only one who achieved infamy in the pages of history. He was a sheriff - yes, that sheriff. And he lived the remainder of his life in disgrace for failing to catch that notorious outlaw. Everywhere he went, people whispered about his tremendous failure - outwitted by a silly mask and some straw.”

There was more rain, and more lightning, as Charles entered the dining room. Anticipating my thoughts, he announced, "No, I didn't steal the mask, but I wish I had. And I'll tell you one thing more. I hope you aren't able to find it, any more than my sheriff ancestor found your hidden outlaw.”

Chester laughed at Bertram's obvious dismay as he carried in a pecan pie that he'd prepared in the cook's quarters, and I began the search for clues.

It was eerie to stand surrounded by empty suits of ancient armor, but it soon paid off. Bertram and I discovered a trail of footprints that led to the stairs.

Your thief came from outside, I told Bertram, who seemed surprised. "But it's a cold, rainy, and windy night. Who would want to leave a cozy castle for an evening like this? I can't be sure, but I didn't notice any of my guests having wet hair.”

Each guest agreed to let us search their room. Bertram searched through all of the bags that Mr. Winfrey had brought, along with his room, but there was no sign of the scarecrow's mask. And while Charles Kincaid had two fine suits hanging in his closet, a search of his room, his suits, and his pockets all failed to turn up the scarecrow's mask.

"Perhaps the king's soldiers have been fooled again," he sneered.

If the thief hadn't traveled upstairs, perhaps he'd headed downstairs? There were no clues on the east staircase, but on the west staircase, my candle illuminated a strange, white scrap, which was wet and slimy.

I tilted my candle to the floor, and discovered. . . a potato peel.

As though having a second dinner, everyone gathered around the table downstairs. I announced I'd determined who'd stolen the scarecrow's mask.
    
    """

    # Example prompt
    prompt = create_basic_cot_prompt(
        suspects=["Charles Kincaid", "Chester", "Mrs. Winfrey", "Mr. Winfrey"],
        mystery_text=mystery_text
    )    
    # Run inference
    result = run_inference(pipe, prompt)
    print(f"Output: {result}")
