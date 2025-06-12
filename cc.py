from deepeval.dataset.golden import Golden
from deepeval.synthesizer import Synthesizer

goldens_to_evolve = [
    Golden(input="What is the capital of France?"),
    Golden(input="Who painted the Mona Lisa?"),
    Golden(input="What is the largest planet in our solar system?"),
    Golden(input="Who wrote 'Romeo and Juliet'?"),
    Golden(input="What is the chemical symbol for gold?"),
    Golden(input="Who discovered penicillin?"),
    Golden(input="What is the tallest mountain in the world?"),
    Golden(input="Who composed the 'Moonlight Sonata'?"),
    Golden(input="What is the largest ocean on Earth?"),
    Golden(input="Who painted 'The Starry Night'?"),
    Golden(input="What is the speed of light in meters per second?"),
    Golden(input="Who wrote 'The Great Gatsby'?"),
]

synthesizer = Synthesizer()

print("Attempting to generate goldens from goldens...")

# This call does not populate synthesizer.synthetic_goldens internally in async mode
evolved_goldens = synthesizer.generate_goldens_from_goldens(
    goldens=goldens_to_evolve, max_goldens_per_golden=1
)

print(f"Generated {len(evolved_goldens)} evolved goldens.")
print(f"Count of goldens in synthesizer: {len(synthesizer.synthetic_goldens)}")
