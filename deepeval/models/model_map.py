model_map = {
    "snli-base": {
        "model_card": "boychaboy/SNLI_roberta-base",
        "entailment_idx": 0,
        "contradiction_idx": 2,
    },
    "snli-large": {
        "model_card": "boychaboy/SNLI_roberta-large",
        "entailment_idx": 0,
        "contradiction_idx": 2,
    },
    "mnli-base": {
        "model_card": "microsoft/deberta-base-mnli",
        "entailment_idx": 2,
        "contradiction_idx": 0,
    },
    "mnli": {
        "model_card": "roberta-large-mnli",
        "entailment_idx": 2,
        "contradiction_idx": 0,
    },
    "anli": {
        "model_card": "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
        "entailment_idx": 0,
        "contradiction_idx": 2,
    },
    "vitc-base": {
        "model_card": "tals/albert-base-vitaminc-mnli",
        "entailment_idx": 0,
        "contradiction_idx": 1,
    },
    "vitc": {
        "model_card": "tals/albert-xlarge-vitaminc-mnli",
        "entailment_idx": 0,
        "contradiction_idx": 1,
    },
    "vitc-only": {
        "model_card": "tals/albert-xlarge-vitaminc",
        "entailment_idx": 0,
        "contradiction_idx": 1,
    },
    # "decomp": 0,
    "vectara-hallucination": {
        "model_card": "vectara/hallucination_evaluation_model",
        "entailment_idx": None,
        "contradiction_idx": None,
    },
}


def card_to_name(card):
    card2name = {v["model_card"]: k for k, v in model_map.items()}
    if card in card2name:
        return card2name[card]
    return card


def name_to_card(name):
    if name in model_map:
        return model_map[name]["model_card"]
    return name


def get_neutral_idx(ent_idx, con_idx):
    return list(set([0, 1, 2]) - set([ent_idx, con_idx]))[0]
