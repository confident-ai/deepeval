# mypy: check_untyped_defs = False
###############################################
# Source: https://github.com/tingofurro/summac
###############################################

import nltk
import numpy as np
import torch
import os
import json
from deepeval import utils as utils_misc


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


class _SummaCImager:
    def __init__(
        self,
        model_name="mnli",
        granularity="paragraph",
        use_cache=True,
        max_doc_sents=100,
        device="cuda",
        **kwargs
    ):
        self.grans = granularity.split("-")

        assert (
            all(
                gran in ["paragraph", "sentence", "document", "2sents", "mixed"]
                for gran in self.grans
            )
            and len(self.grans) <= 2
        ), "Unrecognized `granularity` %s" % (granularity)
        assert (
            model_name in model_map.keys()
        ), "Unrecognized model name: `%s`" % (model_name)

        self.model_name = model_name
        if model_name != "decomp":
            self.model_card = name_to_card(model_name)
            self.entailment_idx = model_map[model_name]["entailment_idx"]
            self.contradiction_idx = model_map[model_name]["contradiction_idx"]
            self.neutral_idx = get_neutral_idx(
                self.entailment_idx, self.contradiction_idx
            )

        self.granularity = granularity
        self.use_cache = use_cache
        self.cache_folder = "/export/share/plaban/summac_cache/"

        self.max_doc_sents = max_doc_sents
        self.max_input_length = 500
        self.device = device
        self.cache = {}
        self.model = None  # Lazy loader

    def load_nli(self):
        if self.model_name == "decomp":
            try:
                from allennlp.predictors.predictor import Predictor
            except ModuleNotFoundError:
                print(
                    "allennlp library is not installed. "
                    "Please install the library by following the instruction from their documentation:"
                    "https://docs.allennlp.org/main/"
                )
            self.model = Predictor.from_path(
                "https://storage.googleapis.com/allennlp-public-models/decomposable-attention-elmo-2020.04.09.tar.gz",
                cuda_device=0,
            )

        else:
            try:
                from transformers import (
                    AutoTokenizer,
                    AutoModelForSequenceClassification,
                )
            except ModuleNotFoundError:
                print(
                    "transformers library is not installed. Run 'pip install transformers'"
                )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_card)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_card
            ).eval()
            self.model.to(self.device)

    def split_sentences(self, text):
        sentences = nltk.tokenize.sent_tokenize(text)
        sentences = [sent for sent in sentences if len(sent) > 10]
        return sentences

    def split_2sents(self, text):
        sentences = nltk.tokenize.sent_tokenize(text)
        sentences = [sent for sent in sentences if len(sent) > 10]
        two_sents = [
            " ".join(sentences[i : (i + 2)]) for i in range(len(sentences))
        ]
        return two_sents

    def split_paragraphs(self, text):
        if text.count("\n\n") > 0:
            paragraphs = [p.strip() for p in text.split("\n\n")]
        else:
            paragraphs = [p.strip() for p in text.split("\n")]
        return [p for p in paragraphs if len(p) > 10]

    def split_text(self, text, granularity="sentence"):
        if granularity == "document":
            return [text]
        elif granularity == "paragraph":
            return self.split_paragraphs(text)
        elif granularity == "sentence":
            return self.split_sentences(text)
        elif granularity == "2sents":
            return self.split_2sents(text)
        elif granularity == "mixed":
            return self.split_sentences(text) + self.split_paragraphs(text)

    def build_image(self, original, generated):
        cache_key = (original, generated)
        if self.use_cache and cache_key in self.cache:
            cached_image = self.cache[cache_key]
            cached_image = cached_image[:, : self.max_doc_sents, :]
            return cached_image

        if len(self.grans) == 1:
            gran_doc, gran_sum = self.grans[0], self.grans[0]
        else:
            gran_doc, gran_sum = self.grans[0], self.grans[1]

        original_chunks = self.split_text(original, granularity=gran_doc)[
            : self.max_doc_sents
        ]
        generated_chunks = self.split_text(generated, granularity=gran_sum)

        N_ori = len(original_chunks)
        N_gen = len(generated_chunks)

        if N_ori == 0 or N_gen == 0:
            return np.zeros((3, 1, 1))
        # assert (N_ori > 0 and N_gen > 0), "One of the inputs has no chunks"

        image = np.zeros((3, N_ori, N_gen))

        if self.model is None:
            self.load_nli()

        dataset = [
            {
                "premise": original_chunks[i],
                "hypothesis": generated_chunks[j],
                "doc_i": i,
                "gen_i": j,
            }
            for i in range(N_ori)
            for j in range(N_gen)
        ]
        for batch in utils_misc.batcher(dataset, batch_size=20):
            if self.model_name == "decomp":
                batch_evids, batch_conts, batch_neuts = [], [], []
                batch_json = [
                    {"premise": d["premise"], "hypothesis": d["hypothesis"]}
                    for d in batch
                ]
                model_outs = self.model.predict_batch_json(batch_json)
                for out in model_outs:
                    probs = out["label_probs"]
                    batch_evids.append(probs[0])
                    batch_conts.append(probs[1])
                    batch_neuts.append(probs[2])

            else:
                batch_prems = [b["premise"] for b in batch]
                batch_hypos = [b["hypothesis"] for b in batch]
                batch_tokens = self.tokenizer.batch_encode_plus(
                    list(zip(batch_prems, batch_hypos)),
                    padding=True,
                    truncation=True,
                    max_length=self.max_input_length,
                    return_tensors="pt",
                    truncation_strategy="only_first",
                )
                batch_tokens = {
                    k: v.to(self.device) for k, v in batch_tokens.items()
                }
                with torch.no_grad():
                    model_outputs = self.model(**batch_tokens)

                batch_probs = torch.nn.functional.softmax(
                    model_outputs["logits"], dim=-1
                )
                batch_evids = batch_probs[:, self.entailment_idx].tolist()
                batch_conts = batch_probs[:, self.contradiction_idx].tolist()
                batch_neuts = batch_probs[:, self.neutral_idx].tolist()

            for b, evid, cont, neut in zip(
                batch, batch_evids, batch_conts, batch_neuts
            ):
                image[0, b["doc_i"], b["gen_i"]] = evid
                image[1, b["doc_i"], b["gen_i"]] = cont
                image[2, b["doc_i"], b["gen_i"]] = neut

        if self.use_cache:
            self.cache[cache_key] = image
        return image

    def get_cache_file(self):
        return os.path.join(
            self.cache_folder,
            "cache_%s_%s.json" % (self.model_name, self.granularity),
        )

    def save_cache(self):
        cache_cp = {"[///]".join(k): v.tolist() for k, v in self.cache.items()}
        with open(self.get_cache_file(), "w") as f:
            json.dump(cache_cp, f)

    def load_cache(self):
        cache_file = self.get_cache_file()
        if os.path.isfile(cache_file):
            with open(cache_file, "r") as f:
                cache_cp = json.load(f)
                self.cache = {
                    tuple(k.split("[///]")): np.array(v)
                    for k, v in cache_cp.items()
                }


class _SummaCConv(torch.nn.Module):
    def __init__(
        self,
        models=["mnli", "anli", "vitc"],
        bins="even50",
        granularity="sentence",
        nli_labels="e",
        device="cuda",
        start_file=None,
        imager_load_cache=True,
        agg="mean",
        norm_histo=False,
        **kwargs
    ):
        # `bins` should be `even%d` or `percentiles`
        assert nli_labels in [
            "e",
            "c",
            "n",
            "ec",
            "en",
            "cn",
            "ecn",
        ], "Unrecognized nli_labels argument %s" % (nli_labels)

        super(SummaCConv, self).__init__()
        self.device = device
        self.models = models

        self.imagers = []
        for model_name in models:
            self.imagers.append(
                SummaCImager(
                    model_name=model_name, granularity=granularity, **kwargs
                )
            )
        if imager_load_cache:
            for imager in self.imagers:
                imager.load_cache()
        assert len(self.imagers) > 0, "Imager names were empty or unrecognized"

        if "even" in bins:
            n_bins = int(bins.replace("even", ""))
            self.bins = list(np.arange(0, 1, 1 / n_bins)) + [1.0]
        elif bins == "percentile":
            self.bins = [
                0.0,
                0.01,
                0.02,
                0.03,
                0.04,
                0.07,
                0.13,
                0.37,
                0.90,
                0.91,
                0.92,
                0.93,
                0.94,
                0.95,
                0.955,
                0.96,
                0.965,
                0.97,
                0.975,
                0.98,
                0.985,
                0.99,
                0.995,
                1.0,
            ]

        self.nli_labels = nli_labels
        self.n_bins = len(self.bins) - 1
        self.norm_histo = norm_histo
        self.n_rows = 10
        self.n_labels = 2
        self.n_depth = len(self.imagers) * len(self.nli_labels)
        self.full_size = self.n_depth * self.n_bins
        if self.norm_histo:
            self.full_size += (
                2  # Will explicitely give the count of originals and generateds
            )

        self.agg = agg

        self.mlp = torch.nn.Linear(self.full_size, 1).to(device)
        self.layer_final = torch.nn.Linear(3, self.n_labels).to(device)

        if start_file is not None:
            print(self.load_state_dict(torch.load(start_file)))

    def build_image(self, original, generated):
        images = [
            imager.build_image(original, generated) for imager in self.imagers
        ]
        image = np.concatenate(images, axis=0)
        return image

    def compute_histogram(self, original=None, generated=None, image=None):
        # Takes the two texts, and generates a (n_rows, 2*n_bins)

        if image is None:
            image = self.build_image(original, generated)

        N_depth, N_ori, N_gen = image.shape

        full_histogram = []
        for i_gen in range(N_gen):
            histos = []

            for i_depth in range(N_depth):
                if (
                    (i_depth % 3 == 0 and "e" in self.nli_labels)
                    or (i_depth % 3 == 1 and "c" in self.nli_labels)
                    or (i_depth % 3 == 2 and "n" in self.nli_labels)
                ):
                    histo, X = np.histogram(
                        image[i_depth, :, i_gen],
                        range=(0, 1),
                        bins=self.bins,
                        density=self.norm_histo,
                    )
                    histos.append(histo)

            if self.norm_histo:
                histos = [[N_ori, N_gen]] + histos
            histogram_row = np.concatenate(histos)
            full_histogram.append(histogram_row)

        n_rows_missing = self.n_rows - len(full_histogram)
        full_histogram += [[0.0] * self.full_size] * n_rows_missing
        full_histogram = full_histogram[: self.n_rows]
        full_histogram = np.array(full_histogram)
        return image, full_histogram

    def forward(self, originals, generateds, images=None):
        if images is not None:
            # In case they've been pre-computed.
            histograms = []
            for image in images:
                _, histogram = self.compute_histogram(image=image)
                histograms.append(histogram)
        else:
            images, histograms = [], []
            for original, generated in zip(originals, generateds):
                image, histogram = self.compute_histogram(
                    original=original, generated=generated
                )
                images.append(image)
                histograms.append(histogram)

        N = len(histograms)
        histograms = torch.FloatTensor(histograms).to(self.device)

        non_zeros = (torch.sum(histograms, dim=-1) != 0.0).long()
        seq_lengths = non_zeros.sum(dim=-1).tolist()

        mlp_outs = self.mlp(histograms).reshape(N, self.n_rows)
        features = []

        for mlp_out, seq_length in zip(mlp_outs, seq_lengths):
            if seq_length > 0:
                Rs = mlp_out[:seq_length]
                if self.agg == "mean":
                    features.append(
                        torch.cat(
                            [
                                torch.mean(Rs).unsqueeze(0),
                                torch.mean(Rs).unsqueeze(0),
                                torch.mean(Rs).unsqueeze(0),
                            ]
                        ).unsqueeze(0)
                    )
                elif self.agg == "min":
                    features.append(
                        torch.cat(
                            [
                                torch.min(Rs).unsqueeze(0),
                                torch.min(Rs).unsqueeze(0),
                                torch.min(Rs).unsqueeze(0),
                            ]
                        ).unsqueeze(0)
                    )
                elif self.agg == "max":
                    features.append(
                        torch.cat(
                            [
                                torch.max(Rs).unsqueeze(0),
                                torch.max(Rs).unsqueeze(0),
                                torch.max(Rs).unsqueeze(0),
                            ]
                        ).unsqueeze(0)
                    )
                elif self.agg == "all":
                    features.append(
                        torch.cat(
                            [
                                torch.min(Rs).unsqueeze(0),
                                torch.mean(Rs).unsqueeze(0),
                                torch.max(Rs).unsqueeze(0),
                            ]
                        ).unsqueeze(0)
                    )
            else:
                features.append(
                    torch.FloatTensor([0.0, 0.0, 0.0]).unsqueeze(0)
                )  # .cuda()
        features = torch.cat(features)
        logits = self.layer_final(features)
        histograms_out = [histogram.cpu().numpy() for histogram in histograms]
        return logits, histograms_out, images

    def save_imager_cache(self):
        for imager in self.imagers:
            imager.save_cache()

    def score(self, originals, generateds, **kwargs):
        with torch.no_grad():
            logits, histograms, images = self.forward(originals, generateds)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            batch_scores = probs[:, 1].tolist()
        return {
            "scores": batch_scores
        }  # , "histograms": histograms, "images": images


class _SummaCZS:
    def __init__(
        self,
        model_name="mnli",
        granularity="paragraph",
        op1="max",
        op2="mean",
        use_ent=True,
        use_con=True,
        imager_load_cache=True,
        device="cuda",
        **kwargs
    ):
        assert op2 in ["min", "mean", "max"], "Unrecognized `op2`"
        assert op1 in ["max", "mean", "min"], "Unrecognized `op1`"

        self.imager = _SummaCImager(
            model_name=model_name,
            granularity=granularity,
            device=device,
            **kwargs
        )
        if imager_load_cache:
            self.imager.load_cache()
        self.op2 = op2
        self.op1 = op1
        self.use_ent = use_ent
        self.use_con = use_con

    def save_imager_cache(self):
        self.imager.save_cache()

    def score_one(self, original, generated):
        image = self.imager.build_image(original, generated)

        ent_scores = np.max(image[0], axis=0)
        co_scores = np.max(image[1], axis=0)
        if self.op1 == "mean":
            ent_scores = np.mean(image[0], axis=0)
            co_scores = np.mean(image[1], axis=0)
        elif self.op1 == "min":
            ent_scores = np.min(image[0], axis=0)
            co_scores = np.min(image[1], axis=0)

        if self.use_ent and self.use_con:
            scores = ent_scores - co_scores
        elif self.use_ent:
            scores = ent_scores
        elif self.use_con:
            scores = 1.0 - co_scores

        final_score = np.mean(scores)
        if self.op2 == "min":
            final_score = np.min(scores)
        elif self.op2 == "max":
            final_score = np.max(scores)

        return {"score": final_score, "image": image}

    def score(self, sources, generateds, **kwargs):
        output = {"scores": [], "images": []}
        for source, gen in zip(sources, generateds):
            score = self.score_one(source, gen)
            output["scores"].append(score["score"])
            output["images"].append(score["image"])
        return output
