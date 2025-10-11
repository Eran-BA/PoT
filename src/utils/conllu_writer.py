"""
CoNLL-U format writer for dependency parsing predictions

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

from typing import List, Optional


def write_conllu(
    path: str,
    tokens: List[List[str]],
    heads_gold: Optional[List[List[int]]] = None,
    deprels_gold: Optional[List[List[str]]] = None,
    heads_pred: Optional[List[List[int]]] = None,
    deprels_pred: Optional[List[List[str]]] = None,
):
    """
    Write predictions to CoNLL-U format for official evaluation.

    If both gold & pred provided, stores predictions as MISC=PredHead=x|PredRel=y

    Args:
        path: Output file path
        tokens: List of sentences, each a list of token strings
        heads_gold: Gold standard head indices (0 = ROOT)
        deprels_gold: Gold standard dependency relations
        heads_pred: Predicted head indices
        deprels_pred: Predicted dependency relations
    """

    def safe(x, default="_"):
        return x if x is not None else default

    with open(path, "w", encoding="utf-8") as f:
        for s_idx, toks in enumerate(tokens):
            f.write(f"# sent_id = {s_idx}\n")
            f.write(f"# text = {' '.join(toks)}\n")

            L = len(toks)
            for i in range(L):
                idx = i + 1
                form = toks[i]

                # Gold annotations (with safe access)
                try:
                    head_g = (
                        str(heads_gold[s_idx][i]) if heads_gold and s_idx < len(heads_gold) else "_"
                    )
                except (IndexError, TypeError):
                    head_g = "_"

                try:
                    rel_g = (
                        deprels_gold[s_idx][i]
                        if deprels_gold and s_idx < len(deprels_gold)
                        else "_"
                    )
                except (IndexError, TypeError):
                    rel_g = "_"

                # Predictions (stored in MISC field)
                try:
                    head_p = (
                        str(heads_pred[s_idx][i])
                        if heads_pred and s_idx < len(heads_pred)
                        else None
                    )
                except (IndexError, TypeError):
                    head_p = None

                try:
                    rel_p = (
                        deprels_pred[s_idx][i]
                        if deprels_pred and s_idx < len(deprels_pred)
                        else None
                    )
                except (IndexError, TypeError):
                    rel_p = None

                misc = []
                if head_p is not None:
                    misc.append(f"PredHead={head_p}")
                if rel_p is not None:
                    misc.append(f"PredRel={rel_p}")
                misc_str = "|".join(misc) if misc else "_"

                # CoNLL-U format: ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC
                f.write(
                    "\t".join(
                        map(
                            str,
                            [
                                idx,
                                form,
                                "_",
                                "_",
                                "_",
                                "_",
                                safe(head_g),
                                safe(rel_g),
                                "_",
                                misc_str,
                            ],
                        )
                    )
                    + "\n"
                )

            f.write("\n")
