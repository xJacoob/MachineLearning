# Results

Train: `polemo2-in/train`. Model/variant selection: `polemo2-in/validation` (macro F1).
Final evaluation (once): `polemo2-in/test` (in-domain) and `polemo2-out/test` (out-of-domain).

## Transformers

| Model | Test set | Val macro F1 | Accuracy | Macro F1 | Notes |
|---|---|---|---|---|---|
| HerBERT (3 epochs) | in-domain | 0.874 | 0.898 | 0.876 | lr 2e-5, MAX_LENGTH=512, 3.5 min (RTX 4060 Ti); val loss rose in epoch 3 |
| HerBERT (3 epochs) | out-of-domain | 0.874 | 0.755 | 0.572 | only 1 "zero" example in out/test, so macro F1 is pessimistic |
| HerBERT (2 epochs), run 1 | in-domain | 0.884 | 0.906 | 0.887 | lower val loss (0.297 vs 0.393), 2.5 min → 2 epochs chosen for all models |
| HerBERT (2 epochs), run 1 | out-of-domain | 0.884 | 0.796 | 0.604 | less overfitting → better generalization |
| **HerBERT (2 epochs), run 2** | in-domain | **0.870** | 0.904 | 0.884 | same settings as run 1; trained together with XLM-R |
| **HerBERT (2 epochs), run 2** | out-of-domain | **0.870** | 0.775 | 0.588 | |
| **XLM-RoBERTa base (2 epochs)** | in-domain | **0.854** | 0.899 | 0.882 | `FacebookAI/xlm-roberta-base`, same settings as HerBERT |
| **XLM-RoBERTa base (2 epochs)** | out-of-domain | **0.854** | 0.785 | 0.625 | slightly better out-of-domain than HerBERT run 2 |

**Run-to-run variance:** two runs of HerBERT with identical settings differ by
0.014 val macro F1 and 2.1 pts out-of-domain accuracy (the classification head
was not seeded before `from_pretrained`; fixed with `set_seed` for future runs).
Differences between HerBERT and XLM-R are of the same size, so a single run
does not show that either model is better.

## Baseline: TF-IDF + Logistic Regression

| Model | Test set | Val macro F1 | Accuracy | Macro F1 | Notes |
|---|---|---|---|---|---|
| TF-IDF + LogReg (default) | in-domain | 0.784 | 0.824 | 0.788 | default TfidfVectorizer, LogisticRegression(max_iter=1000), <1 s on CPU |
| TF-IDF + LogReg (default) | out-of-domain | 0.784 | 0.532 | 0.376 | mixed-class recall only 0.14; macro F1 without "zero" ≈ 0.50 |
| TF-IDF + LogReg (bigrams + sublinear_tf) | in-domain | 0.773 | 0.816 | 0.769 | ngram_range=(1, 2), sublinear_tf=True: worse, collapses to majority class |
| TF-IDF + LogReg (bigrams + sublinear_tf) | out-of-domain | 0.773 | 0.512 | 0.317 | mixed-class recall drops to 0.03 |
| **TF-IDF + LogReg (balanced)** | in-domain | **0.797** | 0.781 | 0.769 | class_weight="balanced"; **selected on validation → official baseline** |
| **TF-IDF + LogReg (balanced)** | out-of-domain | **0.797** | 0.528 | 0.416 | mixed-class recall 0.31 (vs 0.14 default) |

## Summary

| Model | In-domain acc / F1 | Out-of-domain acc / F1 | Accuracy drop |
|---|---|---|---|
| TF-IDF + LogReg (balanced) | 0.781 / 0.769 | 0.528 / 0.416 | −25.3 pts |
| HerBERT (2 epochs, run 2) | 0.904 / 0.884 | 0.775 / 0.588 | −12.9 pts |
| XLM-RoBERTa base (2 epochs) | 0.899 / 0.882 | 0.785 / 0.625 | −11.4 pts |