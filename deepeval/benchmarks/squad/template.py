class SQuADTemplate:

    n_shot_examples = [
        {
            "id": "56bfe7eaa10cfb1400551387",
            "title": "Beyoncé",
            "context": "After Hurricane Katrina in 2005, Beyoncé and Rowland founded the Survivor Foundation to provide transitional housing for victims in the Houston area, to which Beyoncé contributed an initial $250,000. The foundation has since expanded to work with other charities in the city, and also provided relief following Hurricane Ike three years later.",
            "question": "What did Beyonce and Rowland found in 2005?",
            "answers": {
                "text": ["the Survivor Foundation"],
                "answer_start": [61],
            },
        },
        {
            "id": "56d3823659d6e414001465b6",
            "title": "Frédéric_Chopin",
            "context": 'With his health further deteriorating, Chopin desired to have a family member with him. In June 1849 his sister Ludwika came to Paris with her husband and daughter, and in September, supported by a loan from Jane Stirling, he took an apartment at Place Vendôme 12. After 15 October, when his condition took a marked turn for the worse, only a handful of his closest friends remained with him, although Viardot remarked sardonically that "all the grand Parisian ladies considered it de rigueur to faint in his room."',
            "question": "Which family member came to Paris in June 1849?",
            "answers": {"text": ["his sister"], "answer_start": [101]},
        },
        {
            "id": "56d135e0e7d4791d00902016",
            "title": "The_Legend_of_Zelda:_Twilight_Princess",
            "context": "Twilight Princess received the awards for Best Artistic Design, Best Original Score, and Best Use of Sound from IGN for its GameCube version. Both IGN and Nintendo Power gave Twilight Princess the awards for Best Graphics and Best Story. Twilight Princess received Game of the Year awards from GameTrailers, 1UP.com, Electronic Gaming Monthly, Game Informer, Games Radar, GameSpy, Spacey Awards, X-Play and Nintendo Power. It was also given awards for Best Adventure Game from the Game Critics Awards, X-Play, IGN, GameTrailers, 1UP.com, and Nintendo Power. The game was considered the Best Console Game by the Game Critics Awards and GameSpy. The game placed 16th in Official Nintendo Magazine's list of the 100 Greatest Nintendo Games of All Time. IGN ranked the game as the 4th-best Wii game. Nintendo Power ranked the game as the third-best game to be released on a Nintendo system in the 2000s decade.",
            "question": "What award did Game Critics Awards and GameSpy give Twilight Princess?",
            "answers": {"text": ["Best Console Game"], "answer_start": [586]},
        },
        {
            "id": "56ceeb94aab44d1400b88cb3",
            "title": "New_York_City",
            "context": "The city and surrounding area suffered the bulk of the economic damage and largest loss of human life in the aftermath of the September 11, 2001 attacks when 10 of the 19 terrorists associated with Al-Qaeda piloted American Airlines Flight 11 into the North Tower of the World Trade Center and United Airlines Flight 175 into the South Tower of the World Trade Center, and later destroyed them, killing 2,192 civilians, 343 firefighters, and 71 law enforcement officers who were in the towers and in the surrounding area. The rebuilding of the area, has created a new One World Trade Center, and a 9/11 memorial and museum along with other new buildings and infrastructure. The World Trade Center PATH station, which opened on July 19, 1909 as the Hudson Terminal, was also destroyed in the attack. A temporary station was built and opened on November 23, 2003. A permanent station, the World Trade Center Transportation Hub, is currently under construction. The new One World Trade Center is the tallest skyscraper in the Western Hemisphere and the fourth-tallest building in the world by pinnacle height, with its spire reaching a symbolic 1,776 feet (541.3 m) in reference to the year of American independence.",
            "question": "How many firefighters died in the World Trade Center attack?",
            "answers": {"text": ["343"], "answer_start": [420]},
        },
        {
            "id": "56d0875b234ae51400d9c348",
            "title": "Solar_energy",
            "context": "Greenhouses convert solar light to heat, enabling year-round production and the growth (in enclosed environments) of specialty crops and other plants not naturally suited to the local climate. Primitive greenhouses were first used during Roman times to produce cucumbers year-round for the Roman emperor Tiberius. The first modern greenhouses were built in Europe in the 16th century to keep exotic plants brought back from explorations abroad. Greenhouses remain an important part of horticulture today, and plastic transparent materials have also been used to similar effect in polytunnels and row covers.",
            "question": "What do greenhouses do with solar energy?",
            "answers": {
                "text": ["convert solar light to heat"],
                "answer_start": [12],
            },
        },
    ]

    @staticmethod
    def generate_output(input: str, n_shots: int):
        prompt = ""
        for i in range(n_shots):
            prompt += SQuADTemplate.format_question(
                SQuADTemplate.n_shot_examples[i]
            )
        prompt += input
        return prompt

    @staticmethod
    def format_question(data: dict, include_answer=True):
        question = data["question"]
        context = data["context"]
        answer = data["answers"]["text"][0]
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        prompt += ""
        if include_answer:
            prompt += " {}\n\n".format(answer)
        return prompt

    @staticmethod
    def format_output(data: dict):
        return data["answers"]["text"][0]
