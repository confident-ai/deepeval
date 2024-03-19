class SynthesizerTemplate:
    @staticmethod
    def generate_synthetic_data(context, max_goldens_per_context):
        return f"""I want you act as a copywriter. Based on the given context, which is list of strings, please generate a list of JSON objects with a `input` key.
The `input` can either be a question or a statement that can be addressed by the given context.

**
IMPORTANT: Please make sure to only return in JSON format, with the 'data' key as a list of JSON objects.
You MUST TRY to generate {max_goldens_per_context} data points, unless the `input` is getting repetitive.

Example context: ["Einstein won the Nobel Prize for his discovery of penicillin.", "Einstein won the Nobel Prize in 1968."]
Example max goldens per context: 2
Example JSON:
{{
    "data": [
        {{
            "input": "What was Einstein known for?"
        }},
        {{
            "input": "Einstein was a smart guy huh"
        }}
    ]  
}}


You should NOT incorporate any prior knowledge you have and take each context at face value.
You MUST include at least one statement as the input.
`input` MUST be a STRING.
You MUST TRY to generate {max_goldens_per_context} data points, unless the `input` is getting repetitive.
**

Max Goldens Per Context:
{max_goldens_per_context}

Context:
{context}

JSON:
"""


######################################################################################################
##### Approach similar to https://github.com/nlpxucan/WizardLM/blob/main/Evol_Instruct/depth.py ######
######################################################################################################

# generate_constraints_prompt
# "Please add one more constraints/requirements into #The Given Prompt#'"

# generate_deepen_prompt
# "If #The Given Prompt# contains inquiries about certain issues, the depth and breadth of the inquiry can be increased."

# generate_concretizing_prompt
#  "Please replace general concepts with more specific concepts."

# generate_reasoning_prompt
# "If #The Given Prompt# can be solved with just a few simple thinking processes, you can rewrite it to explicitly request multiple-step reasoning."

depth_base_instruction = """I want you to act as an input rewriter.
Your object is the rewrite a given `input` and must be factually correct according to the supporting information in `context`.
You MUST complicate the given `input` using the following method:"""


class EvolutionTemplate:
    @staticmethod
    def name_to_be_decided_evolution(input, context):
        return (
            depth_base_instruction
            + f"""
    1. Using information from the provided `context`, complicate the `input` by replacing general concepts/inquiries with more specific ones.
    2. Using information help from the provided `context`, increase the depth and breadth of the `input`.

**
EXAMPLES
Example context:
Rainforests are home to over half of the world's plant and animal species, making them key to maintaining global biodiversity. The variety of life found in these ecosystems contributes to genetic diversity, which is crucial for adaptation and survival amid changing environmental conditions. This biodiversity also supports ecosystem resilience, enabling forests to recover from disturbances.
The biodiversity in rainforests plays a significant role in human well-being, providing essential services such as air and water purification, disease control, and pollination of crops. Additionally, many medicines are derived from rainforest plants, highlighting the importance of these ecosystems for medical research and healthcare.
Example input: 
Why is the biodiversity of rainforests important?
Example rewritten input:
How does the extensive biodiversity found in rainforests, encompassing over half of the world's plant and animal species, contribute to global biodiversity maintenance, and what role does this diversity play in enhancing ecosystem resilience, human health through disease control, crop pollination, and the development of medicines derived from rainforest plants?

--------------------------

Example context:
Bees play a critical role in pollinating flowering plants, including many fruits and vegetables, contributing to the diversity of plant life and the production of crops. Their activity supports the growth of trees, flowers, and other plants, which serve as food and shelter for numerous animals, thus maintaining ecosystem balance.
Beyond their impact on food crops, bees contribute to wild plant growth by pollinating a wide range of plants outside of agricultural settings. This pollination is vital for the reproduction of many plants, affecting entire ecosystems' health and sustainability.
Example input: 
What is the role of bees in ecosystems?
Example rewritten input:
How do bees, through their pollination of flowering plants, including a multitude of fruits and vegetables, significantly influence the diversity of plant life and agricultural productivity, and in what ways do their activities extend beyond agricultural settings to support the growth of trees, flowers, and other plants, thereby providing essential resources for various animal species and contributing to the overall balance and sustainability of ecosystems?

--------------------------

Example context:
Solar power generation relies on photovoltaic cells to convert sunlight into electricity. These cells are made of materials that exhibit the photovoltaic effect, which occurs when light photons are absorbed by the material, causing the generation of electrical current.
Solar panels, composed of many photovoltaic cells, collect sunlight and convert it into electrical power. This energy can then be used directly or stored in batteries for later use, providing a renewable and sustainable source of power with minimal environmental impact.
Example input: 
What are the principles behind solar power generation?
Example rewritten input:
How do photovoltaic cells work to convert sunlight into electrical power, and what role do solar panels play in this process, including energy storage for sustainable use?
**

Input:
{input}

Context:
{context}

Rewritten Input:
"""
        )

    @staticmethod
    def second_name_to_be_decided_evolution(input, context):
        return (
            depth_base_instruction
            + f"""
    1. Using information from the provided `context`, complicate the `input` by adding at least one more constraints/requirements.
    2. Using information from the provided `context`, complicate the `input` by requiring multi-step reasoning.

**
EXAMPLES
Example context:
Rainforests are home to over half of the world's plant and animal species, making them key to maintaining global biodiversity. The variety of life found in these ecosystems contributes to genetic diversity, which is crucial for adaptation and survival amid changing environmental conditions. This biodiversity also supports ecosystem resilience, enabling forests to recover from disturbances.
The biodiversity in rainforests plays a significant role in human well-being, providing essential services such as air and water purification, disease control, and pollination of crops. Additionally, many medicines are derived from rainforest plants, highlighting the importance of these ecosystems for medical research and healthcare.
Example input: 
Why is the biodiversity of rainforests important?
Example rewritten input:
How does the biodiversity of rainforests contribute to ecosystem resilience and recovery from disturbances, and in what ways does it impact human well-being through services such as air and water purification, disease control, and crop pollination?

--------------------------

Example context:
Bees play a critical role in pollinating flowering plants, including many fruits and vegetables, contributing to the diversity of plant life and the production of crops. Their activity supports the growth of trees, flowers, and other plants, which serve as food and shelter for numerous animals, thus maintaining ecosystem balance.
Beyond their impact on food crops, bees contribute to wild plant growth by pollinating a wide range of plants outside of agricultural settings. This pollination is vital for the reproduction of many plants, affecting entire ecosystems' health and sustainability.
Example input: 
What is the role of bees in ecosystems?
Example rewritten input:
Considering the pivotal role bees play in pollinating both agricultural crops and wild plants, thereby contributing to the diversity of plant life and supporting the foundation of food chains, analyze how bees influence the growth and sustainability of various ecosystems.

--------------------------

Example context:
Solar power generation relies on photovoltaic cells to convert sunlight into electricity. These cells are made of materials that exhibit the photovoltaic effect, which occurs when light photons are absorbed by the material, causing the generation of electrical current.
Solar panels, composed of many photovoltaic cells, collect sunlight and convert it into electrical power. This energy can then be used directly or stored in batteries for later use, providing a renewable and sustainable source of power with minimal environmental impact.
Example input: 
What are the principles behind solar power generation?
Example rewritten input:
Examine the significance of rainforest biodiversity in sustaining ecosystem resilience and providing essential services such as disease control and crop pollination, alongside its critical role in medical research and the development of new medicines. Consider the broader implications of biodiversity loss on global ecological balance and human health.
**

Context:
{context}

Input:
{input}

Rewritten Input:
"""
        )
