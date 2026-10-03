import nltk
import sacrebleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from nltk.translate import meteor_score

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("omw-1.4")

class Evaluator:
    def __init__(self):
        pass

    @staticmethod
    def evaluate_metrics(references, candidates, lang="en"):
        assert len(references) == len(candidates), "len(refs) doesn't equal to len(candidates)"

        bleu = sacrebleu.corpus_bleu(candidates, [references]).score/100

        meteor_scores = [
            meteor_score.single_meteor_score(
                nltk.word_tokenize(ref),
                nltk.word_tokenize(cand)
            )
            for ref, cand in zip(references, candidates)
        ]
        meteor = sum(meteor_scores) / len(meteor_scores)

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        rouge1, rouge2, rougeL = [], [], []
        for ref, cand in zip(references, candidates):
            scores = scorer.score(ref, cand)
            rouge1.append(scores["rouge1"].fmeasure)
            rouge2.append(scores["rouge2"].fmeasure)
            rougeL.append(scores["rougeL"].fmeasure)
        rouge1 = sum(rouge1) / len(rouge1)
        rouge2 = sum(rouge2) / len(rouge2)
        rougeL = sum(rougeL) / len(rougeL)

        P, R, F1 = bert_score(candidates, references, lang=lang)
        bert_f1 = float(F1.mean())

        return {
            "BLEU": bleu,
            "METEOR": meteor,
            "ROUGE-1": rouge1,
            "ROUGE-2": rouge2,
            "ROUGE-L": rougeL,
            "BERTScore_F1": bert_f1
        }


if __name__ == "__main__":
    refs = ["The patient was prescribed aspirin and advised to rest."]
    cands = ["The patient received aspirin and was told to rest."]

    results = Evaluator.evaluate_metrics(refs, cands)  # 静态方法直接类调用
    # for k, v in results.items():
    #     print(f"{k}: {v:.4f}")

