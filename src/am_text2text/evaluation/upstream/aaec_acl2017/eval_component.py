"""AAEC component evaluation.

Python 3 reimplementation of the original Python 2 AAEC evaluation code from
UKPLab/acl2017-neural_end2end_am, released with Eger et al.,
"Neural End-to-End Learning for Computational Argumentation Mining"
(ACL 2017).

Paper: https://aclanthology.org/P17-1002/
GitHub: https://github.com/UKPLab/acl2017-neural_end2end_am
Upstream code license: Apache License 2.0
"""

import sys

from am_text2text.evaluation.upstream.aaec_acl2017.docReader import readDocsFine2 as readDocsFine


def extractComponents(lst, types):
    items = {}
    index = 0
    for ic, comp in enumerate(lst):
        if types[ic] is not None:
            items[index] = (types[ic], len(comp))
        index += len(comp)
    return items


def checkApproxMatch(a, b, ratio=0.999):
    if a[1][0] != b[1][0]:
        return False
    a_tok = set(range(a[0], a[0] + a[1][1]))
    b_tok = set(range(b[0], b[0] + b[1][1]))
    overlap = float(len(a_tok.intersection(b_tok)))
    return overlap / max(len(a_tok), len(b_tok)) > ratio


def getTP_approx_simple(pred, truth, ratio=0.9999):
    tp = 0
    fp = 0
    fn = 0
    for x in pred:
        if not any(checkApproxMatch((x, pred[x]), (y, truth[y]), ratio=ratio) for y in truth):
            fp += 1
    for x in truth:
        if any(checkApproxMatch((x, truth[x]), (y, pred[y]), ratio=ratio) for y in pred):
            tp += 1
        else:
            fn += 1
    return tp, fp, fn


def compute_metrics_component(pred_file, truth_file, ratio=0.999):
    predDocs, argTypesDocs = readDocsFine(pred_file, 2)
    truthDocs, argTypesDocsTruth = readDocsFine(truth_file, 2)

    tps = 0
    fps = 0
    fns = 0
    for idoc, doc in enumerate(predDocs):
        if len(doc) != len(argTypesDocs[idoc]):
            sys.stderr.write(f"PROBLEM in doc {idoc}\n")
        try:
            pred_c = extractComponents(doc, argTypesDocs[idoc])
            truth_c = extractComponents(truthDocs[idoc], argTypesDocsTruth[idoc])
        except IndexError:
            sys.stderr.write(f"ERROR in doc {idoc}\n")
            continue
        tp, fp, fn = getTP_approx_simple(pred_c, truth_c, ratio=ratio)
        tps += tp
        fps += fp
        fns += fn

    f1 = 2 * tps * 1.0 / max((2 * tps + fps + fns), 1)
    recall = tps / max((tps + fns), 1)
    precision = tps / max((tps + fps), 1)
    return precision, recall, f1
