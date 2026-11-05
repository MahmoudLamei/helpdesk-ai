# EDA: Bitext customer support dataset

## Overview
- 26,872 rows, 5 columns
- Each row is a customer support ticket
- Columns:
    - `flags`: a combination of tags that the meaning of can be found in the dataset README
    - `instruction`: the query of the customer
    - `category`: the category of this query eg. ORDER
    - `intent`: more specific eg. cancel_order
    - `response`: the response from the support

## Duplicates
- What I saw:
    2,237 exact duplicated messages (~8.3%).
    All duplicates share the same category and intent, so there is no label conflict.
    No missing values in any column.
    Exact matching misses near-duplicates that only differ in case or punctuation
    ("Cancel my order" vs "cancel my order!"). After normalizing (lowercase, letters only,
    single spaces), 15.3% of test messages also appeared in the training set.
- What I'll do about it:
    Drop duplicates on a normalized `instruction` key before splitting (`clean.py`), so the
    same text can't appear in both train and test.
    3,105 rows removed, 23,767 remain.
- Removed per category: ORDER 1,001, ACCOUNT 714, REFUND 441, INVOICE 424, DELIVERY 366,
  CONTACT 61, SHIPPING 24, CANCEL 23, PAYMENT 22, SUBSCRIPTION 19, FEEDBACK 10.
- Side effect: dedup is not uniform. `cancel_order` went from 998 to 409 rows (-59%) because its
  messages were very repetitive. It is now the smallest intent (82 test rows vs ~150-200 for most),
  so cleaning created a new imbalance.

## Class balance
- What I saw:
    ACCOUNT for example has ~6000 rows, however, SUBSCRIPTION and CANCEL have under 1000 each.
    At intent level after dedup, `cancel_order` is the smallest class (see Duplicates).
- What I'll do about it:
    Stratified split (stratify=y), so the small classes appear in the test set in their real proportion.
    Macro-F1 as the main metric, because accuracy would hide bad performance on CANCEL and SUBSCRIPTION.

## Placeholders ({{Order Number}} etc.)
- What I saw:
    The messages has placeholders instead of real values
- What I'll do about it:
    maybe it's good for the model to see the placeholder instead of a weird sentence where it would expect a number.
    But real tickets would have actual numbers that I can change for the placeholder so the numbers don't confuse the agent.

## Typos
- What I saw:
    there are typos in the messages those are there to make the model robust to real tickets.
    Error analysis (below) shows typos are the main cause of the word model's mistakes.
- What I'll do about it:
    Leave the data as is, and use character n-grams so the model can handle misspellings.

## Label choice: category vs. intent
- Decision and why:
    Category would give a cleaner and less classes, but intent is more precise. Intent would be a better choice. We can baseline on category, then intent.
- Result: category is too easy to compare models (macro-F1 0.998 for the baseline). Intent
  (27 classes) shows real differences, so all experiments below use intent.

## Flags
- What I saw:
    Each row has letter tags for the type of language variation it contains.
    Full list: https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset
    Most relevant here: Z = typos, Q = colloquial ("can u activ8"), W = offensive language.
- What I'll do about it:
    Not a model input, since real tickets won't come with flags.
    Use them for evaluation: compare macro-F1 on rows with Z or Q to the overall score, to check robustness.

## Experiments (tracked in MLflow, experiment `helpdesk-baseline`)
Setup: TF-IDF + LogisticRegression, normalized dedup, stratified 80/20 split, seed 42.
Each intent run changes one setting from the baseline.

| Label | Change | macro-F1 | `cancel_order` recall | `get_invoice` recall |
|---|---|---|---|---|
| category | baseline | 0.9983 | - | - |
| intent | baseline (word, C=1) | 0.9875 | 0.88 | 0.91 |
| intent | word bigrams (`ngram_max=2`) | 0.9879 | 0.89 | 0.90 |
| intent | `class_weight="balanced"` | 0.9887 | 0.93 | 0.91 |
| intent | `C=10` | 0.9924 | 0.94 | 0.94 |
| intent | char n-grams (`char_wb`, 1-5) | **0.9950** | **0.96** | **0.97** |

What each result means:
- **Bigrams: no effect.** The confused intents were not caused by missing word pairs.
- **balanced: helps only the rare class.** `cancel_order` (smallest intent) improved, `get_invoice`
  (not rare) did not. It fixes imbalance, nothing else.
- **C=10: helps all confused pairs.** At C=1 the regularization shrank the single word that separates
  the pairs (cancel/change, get/check) too much. Less regularization lets the model trust it.
- **char n-grams: best.** Misspellings like "canmcel" still share character pieces with "cancel",
  so the signal survives. Errors on the test set went from 51 (word) to 21 (char).
  Cost: many more features, so a bigger model and slower training. That will matter for serving.

Earlier, on category: no dedup 0.9969, normalized dedup 0.9983. This difference is noise
(a few messages out of ~5,000, and the test set itself changed). Duplicates were not the reason
the score is high: category is just easy on this templated data.

## Error analysis (intent, word baseline)
Most frequent confusions (true -> predicted):
- `get_invoice` -> `check_invoice` (13)
- `cancel_order` -> `change_order` (9)
- `create_account` -> `delete_account` (6)
- `registration_problems` -> `create_account` (4)

What the misclassified messages look like:
- **cancel_order:** almost all are misspellings of "cancel": "cancelong", "cancellng", "cance",
  "canmcel", "casncel". Without the word "cancel", the model falls back on "order" and picks
  change_order. One exception, "I can't aford order", has no cancel-like word at all. It needs
  understanding, not spelling, which a classic model can't do.
- **get_invoice:** also mostly typos ("geting", "downlolad", "semd", "gt") and the word "bill"
  instead of "invoice". Not ambiguous, as I first guessed.
- **create_account:** messages with no action verb ("I want a pro account",
  "I want a {{Account Type}} account") or a typo in the verb ("oen a gold account").

Conclusion: the word model's main weakness is spelling, which is why char n-grams fixed most of it.
With char n-grams, the top confusion is still `get_invoice` -> `check_invoice` (5).

## MLflow notes
- A run that crashed before `log_params` shows as FAILED with no params, so it is impossible to know what it was trying. Log settings as early as possible.
- `mlflow.sklearn.autolog()` logs dozens of model params, but not my own choices (dedup, label),not the test macro-F1, and it skips dataset logging because `X` is a pandas Series.
- Models are saved with pickle by default, and loading a pickle can execute code. Only load models I created; consider the `skops` format for serving.

## Open questions / next steps
- All model choices above were made by looking at the test set. Add cross-validation on the training set to choose settings, and use the test set only once for the final number.
- Combine char n-grams with C=10 and `balanced`.
- Check robustness by flag: macro-F1 on rows with Z (typos) or Q (colloquial) vs overall.
