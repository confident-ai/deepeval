# Evaluating LLMs For Lawyers

In this section, we will discuss the evaluation of Large Language Models (LLMs) for legal applications. We will cover various aspects such as professional responsibility, legal liability, competence, and the evaluation of LLM output. This will provide a comprehensive understanding of how LLMs can be utilized in the legal field and the considerations that need to be taken into account.


1. Professional and Ethical Responsibility:
   - Professional Responsibility: Assessing the ability of the LLM to adhere to professional standards and ethical guidelines expected of a legal professional.
   - Ethical Considerations: Evaluating if the LLM meets a lawyer’s broader ethical obligations to the client, court, and society.

2. Legal Competence and Expertise:
   - Competence: Evaluating the LLM based on its level of expertise, ranging from the knowledge expected of a graduate to a partner.
   - Context Sensitivity: Assessing the LLM's ability to distinguish between black letter law (technical) vs. implicit (industry practice) or jurisdiction variances.

3. Legal Liability:
   - Liability Risks: Understanding the potential liability risks that may arise from relying on the LLM, particularly in high-stakes vs. low stakes scenarios.

4. Adaptability and Currentness:
   - Real-time adaptability: Checking if the LLM is up to date with new cases, policy changes, and public comments, and its ability to adapt its knowledge in real-time.

5. Evaluation of LLM Output:
   - Evaluating the quality, accuracy, and reliability of the output produced by the LLM in the legal context.

When analyzing for legal liability, we found the following prompt to work well with GPT-4. 

```python
import openai

openai.api_key = "xxx"

def get_indemnity_table(query: str):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert legal advisor."},
            {"role": "user", "content": query + """
For the given legal advice, please provide a detailed analysis. 
1. Identify each potential legal liability.
2. For each liability, list any potential ambiguities.
3. Provide a table in Github-flavored markdown summarizing risk level, liability, its ambiguities, and detailed associated penalties including amount of dollar damage and potential jail time if applicable for the lawyer giving the advice if they get it wrong.
Only return the table.
Table:"""}
        ]
    )
    markdown_content = response['choices'][0]['message']['content']
    return markdown_content

```

For much of this evaluation, we credit the incredible work done by LegalBench team.

```
@misc{guha2023legalbench,
      title={LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models}, 
      author={Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
      year={2023},
      eprint={2308.11462},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
@article{koreeda2021contractnli,
  title={ContractNLI: A dataset for document-level natural language inference for contracts},
  author={Koreeda, Yuta and Manning, Christopher D},
  journal={arXiv preprint arXiv:2110.01799},
  year={2021}
}
@article{hendrycks2021cuad,
  title={Cuad: An expert-annotated nlp dataset for legal contract review},
  author={Hendrycks, Dan and Burns, Collin and Chen, Anya and Ball, Spencer},
  journal={arXiv preprint arXiv:2103.06268},
  year={2021}
}
@article{wang2023maud,
  title={MAUD: An Expert-Annotated Legal NLP Dataset for Merger Agreement Understanding},
  author={Wang, Steven H and Scardigli, Antoine and Tang, Leonard and Chen, Wei and Levkin, Dimitry and Chen, Anya and Ball, Spencer and Woodside, Thomas and Zhang, Oliver and Hendrycks, Dan},
  journal={arXiv preprint arXiv:2301.00876},
  year={2023}
}
@inproceedings{wilson2016creation,
  title={The creation and analysis of a website privacy policy corpus},
  author={Wilson, Shomir and Schaub, Florian and Dara, Aswarth Abhilash and Liu, Frederick and Cherivirala, Sushain and Leon, Pedro Giovanni and Andersen, Mads Schaarup and Zimmeck, Sebastian and Sathyendra, Kanthashree Mysore and Russell, N Cameron and others},
  booktitle={Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={1330--1340},
  year={2016}
}
@inproceedings{zheng2021does,
  title={When does pretraining help? assessing self-supervised learning for law and the casehold dataset of 53,000+ legal holdings},
  author={Zheng, Lucia and Guha, Neel and Anderson, Brandon R and Henderson, Peter and Ho, Daniel E},
  booktitle={Proceedings of the eighteenth international conference on artificial intelligence and law},
  pages={159--168},
  year={2021}
}
@article{zimmeck2019maps,
  title={Maps: Scaling privacy compliance analysis to a million apps},
  author={Zimmeck, Sebastian and Story, Peter and Smullen, Daniel and Ravichander, Abhilasha and Wang, Ziqi and Reidenberg, Joel R and Russell, N Cameron and Sadeh, Norman},
  journal={Proc. Priv. Enhancing Tech.},
  volume={2019},
  pages={66},
  year={2019}
}
@article{ravichander2019question,
  title={Question answering for privacy policies: Combining computational and legal perspectives},
  author={Ravichander, Abhilasha and Black, Alan W and Wilson, Shomir and Norton, Thomas and Sadeh, Norman},
  journal={arXiv preprint arXiv:1911.00841},
  year={2019}
}
@article{holzenberger2021factoring,
  title={Factoring statutory reasoning as language understanding challenges},
  author={Holzenberger, Nils and Van Durme, Benjamin},
  journal={arXiv preprint arXiv:2105.07903},
  year={2021}
}
@article{lippi2019claudette,
  title={CLAUDETTE: an automated detector of potentially unfair clauses in online terms of service},
  author={Lippi, Marco and Pa{\l}ka, Przemys{\l}aw and Contissa, Giuseppe and Lagioia, Francesca and Micklitz, Hans-Wolfgang and Sartor, Giovanni and Torroni, Paolo},
  journal={Artificial Intelligence and Law},
  volume={27},
  pages={117--139},
  year={2019},
  publisher={Springer}
}
```

