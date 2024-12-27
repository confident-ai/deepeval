import re


class LAMBADATemplate:

    n_shot_examples = [
        {
            "text": "her pay for the evening was almost double that of the wait staff and although that might not seem like a lot to some people , it was a small fortune to claire . after loading her final tray for a server , claire went to the restroom to freshen up and begin preparations for being loaded into the cake . pam had a couple of young men from college who assisted her into the cake . brian and max were a lot of fun and always made her laugh as they hoisted her up to the top of the cake"
        },
        {
            "text": "`` nineteen , '' she said , and he loosed a breath that could have been sadness or relief or maybe both , and told her that made her magic even more impressive . she debated saying that he would be less impressed once he learned of her nickname for him , but winked at him instead . rowan was frowning when she caught up to him , but said nothing . as they walked away , gavriel murmured , `` good luck , rowan"
        },
        {
            "text": "my assessment of being dead before lunch was n't too far off base . irritably , ezra shook his head , stalking toward `` our '' mat . `` what kind of training have you had ? '' i gulped , and hurried to catch up . `` uh , none . '' `` perfect , '' he muttered , facing me on the mat"
        },
        {
            "text": "` just in case there 's trouble , ' he grunted to sparhawk before the party left the chapterhouse . the day was cold and raw the sky was leaden , and a chill wind whistled through the streets of cimmura as vanion led them towards the palace . there were few people abroad in the streets . sparhawk could not be sure if the citizens were staying inside because of the weather or because some rumours had leaked out about the possibility of trouble"
        },
        {
            "text": "they are racially mixed and all have their mbas , but some of them have other traits i appreciate , as well . but enough of that . where did you get the name arrow ? '' arrow had recovered her poise . she said , `` my mother was an olympic archer . i guess she hoped she would hit a bull 's - eye with me , just as she does with her other arrows"
        },
    ]

    @staticmethod
    def generate_output(input: str, n_shots: int):
        prompt = ""
        for i in range(n_shots):
            prompt += LAMBADATemplate.format_question(
                LAMBADATemplate.n_shot_examples[i]
            )
        prompt += input
        return prompt

    @staticmethod
    def format_question(data: dict, include_answer: bool = True):
        text: str = data["text"]

        # Find last sentence
        match = re.search(r'(?s)(.*[.!?]["\']?)\s*([\n\s\S]*)', text.strip())
        everything_before_last = match.group(1).strip()
        last_sentence = match.group(2).strip()

        # Find last word
        last_word_match = re.search(r"\b\w+\b(?=[^\w]*$)", last_sentence)
        last_sentence_without_last_word = last_sentence[
            : last_word_match.start()
        ].rstrip()
        last_word = last_word_match.group(0)

        # Construct Input Prompt
        prompt = f"Context: {everything_before_last}\nTarget Sentence: {last_sentence_without_last_word} ____ \nTarget Word:"
        if include_answer == True:
            prompt += f" {last_word}\n\n"
        return prompt

    @staticmethod
    def format_answer(data: dict):
        text: str = data["text"]
        match = re.search(r'(?s)(.*[.!?]["\']?)\s*([\n\s\S]*)', text.strip())
        last_sentence = match.group(2).strip()
        text: str = data["text"]
        last_word_match = re.search(r"\b\w+\b(?=[^\w]*$)", last_sentence)
        last_word = last_word_match.group(0)
        return last_word
