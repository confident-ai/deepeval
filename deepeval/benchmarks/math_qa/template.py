class MathQATemplate:

    n_shot_examples = [
        {
            "Problem": "the banker ' s gain of a certain sum due 3 years hence at 10 % per annum is rs . 36 . what is the present worth ?",
            "Rationale": '"explanation : t = 3 years r = 10 % td = ( bg × 100 ) / tr = ( 36 × 100 ) / ( 3 × 10 ) = 12 × 10 = rs . 120 td = ( pw × tr ) / 100 ⇒ 120 = ( pw × 3 × 10 ) / 100 ⇒ 1200 = pw × 3 pw = 1200 / 3 = rs . 400 answer : option a"',
            "options": "a ) rs . 400 , b ) rs . 300 , c ) rs . 500 , d ) rs . 350 , e ) none of these",
            "correct": "a",
            "annotated_formula": "divide(multiply(const_100, divide(multiply(36, const_100), multiply(3, 10))), multiply(3, 10))",
            "linear_formula": "multiply(n2,const_100)|multiply(n0,n1)|divide(#0,#1)|multiply(#2,const_100)|divide(#3,#1)|",
            "category": "gain",
        },
        {
            "Problem": "average age of students of an adult school is 40 years . 120 new students whose average age is 32 years joined the school . as a result the average age is decreased by 4 years . find the number of students of the school after joining of the new students .",
            "Rationale": '"explanation : let the original no . of students be x . according to situation , 40 x + 120 * 32 = ( x + 120 ) 36 ⇒ x = 120 so , required no . of students after joining the new students = x + 120 = 240 . answer : d"',
            "options": "a ) 1200 , b ) 120 , c ) 360 , d ) 240 , e ) none of these",
            "correct": "d",
            "annotated_formula": "multiply(divide(subtract(multiply(add(32, 4), 120), multiply(120, 32)), subtract(40, add(32, 4))), 4)",
            "linear_formula": "add(n2,n3)|multiply(n1,n2)|multiply(n1,#0)|subtract(n0,#0)|subtract(#2,#1)|divide(#4,#3)|multiply(n3,#5)|",
            "category": "general",
        },
        {
            "Problem": "sophia finished 2 / 3 of a book . she calculated that she finished 90 more pages than she has yet to read . how long is her book ?",
            "Rationale": "let xx be the total number of pages in the book , then she finished 23 ⋅ x 23 ⋅ x pages . then she has x − 23 ⋅ x = 13 ⋅ xx − 23 ⋅ x = 13 ⋅ x pages left . 23 ⋅ x − 13 ⋅ x = 9023 ⋅ x − 13 ⋅ x = 90 13 ⋅ x = 9013 ⋅ x = 90 x = 270 x = 270 so the book is 270 pages long . answer : b",
            "options": "a ) 229 , b ) 270 , c ) 877 , d ) 266 , e ) 281",
            "correct": "b",
            "annotated_formula": "divide(90, subtract(const_1, divide(2, 3)))",
            "linear_formula": "divide(n0,n1)|subtract(const_1,#0)|divide(n2,#1)",
            "category": "general",
        },
        {
            "Problem": "120 is what percent of 50 ?",
            "Rationale": '"50 * x = 120 - - > x = 2.4 - - > 2.4 expressed as percent is 240 % . answer : b ."',
            "options": "a ) 5 % , b ) 240 % , c ) 50 % , d ) 2 % , e ) 500 %",
            "correct": "b",
            "annotated_formula": "multiply(divide(120, 50), const_100)",
            "linear_formula": "divide(n0,n1)|multiply(#0,const_100)|",
            "category": "gain",
        },
        {
            "Problem": "there are 10 girls and 20 boys in a classroom . what is the ratio of girls to boys ?",
            "Rationale": "if girls is 10 and boys is 20 , then 10 / 20 . so ratio of girls to boys is = 10 / 20 = 1 / 2 answer : a",
            "options": "a ) 1 / 2 , b ) 1 / 3 , c ) 1 / 5 , d ) 10 / 30 , e ) 2 / 5",
            "correct": "a",
            "annotated_formula": "divide(10, 20)",
            "linear_formula": "divide(n0,n1)",
            "category": "other",
        },
    ]

    @staticmethod
    def generate_output(input: str, n_shots: int):
        prompt = ""
        for i in range(n_shots):
            prompt += MathQATemplate.format_question(
                MathQATemplate.n_shot_examples[i]
            )
        prompt += input
        return prompt

    @staticmethod
    def format_question(data: dict, include_answer=True):
        question = data["Problem"]
        correct = data["correct"]
        options: str = data["options"]
        formatted_options = "\n".join(options.split(", "))
        prompt = f"Question: {question}\n{formatted_options}\nAnswer:"
        prompt += ""
        if include_answer:
            prompt += " {}\n\n".format(correct)
        return prompt

    @staticmethod
    def format_output(data: dict):
        return data["correct"]
