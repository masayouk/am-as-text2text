"""AAEC relation evaluation.

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


def findLastMajorClaim(lst, types):
    index = 0
    c = 0
    for ix, x in enumerate(types):
        if x == "B-MajorClaim":
            index = c
        c += len(lst[ix])
    return index


def extractRelations(lst, types):
    items = {}
    lmc = findLastMajorClaim(lst, types)
    index = 0
    for ic, comp in enumerate(lst):
        if types[ic] == "B-MajorClaim":
            items[index] = (None, 0)
        elif types[ic] == "B-Claim":
            stance = comp[0].split("\t")[2].split(":")[-1]
            items[index] = ("B-Claim:" + stance, lmc)
        elif types[ic] == "B-Premise":
            parts = comp[0].split("\t")[2].split(":")
            if len(parts) == 3:
                stance = parts[-1]
                rel = parts[1]
                items[index] = ("B-Premise:" + stance, int(rel) - 1)
        index += len(comp)
    return items


def extractComponents(lst, types):
    items = {}
    index = 0
    for ic, comp in enumerate(lst):
        if types[ic] is not None:
            items[index] = (types[ic], len(comp))
        index += len(comp)
    return items


def checkApproxMatch(a, b, typeRequirement=True):
    if a[1][1] is None or b[1][1] is None:
        return a[1][1] == b[1][1]
    if typeRequirement and a[1][0] != b[1][0]:
        return False
    a_tok = set(range(a[0], a[0] + a[1][1]))
    b_tok = set(range(b[0], b[0] + b[1][1]))
    return a_tok == b_tok


def computeF1_relations_approx(pred, truth, pred_c, truth_c):
    tp = 0
    fp = 0
    fn = 0

    for x in truth:
        found = False
        source_x = truth_c[x]
        t_x = truth[x][-1]
        target_x = truth_c[t_x] if t_x != 0 else (None, None)

        for y in pred:
            source_y = pred_c[y]
            t_y = pred[y][-1]
            if t_y != 0:
                target_y = pred_c.get(t_y)
                if target_y is None:
                    continue
            else:
                target_y = (None, None)

            if (
                truth[x][0] == pred[y][0]
                and checkApproxMatch((x, source_x), (y, source_y), typeRequirement=False)
                and checkApproxMatch((t_x, target_x), (t_y, target_y), typeRequirement=False)
            ):
                tp += 1
                found = True
                break
        if not found:
            fn += 1

    for y in pred:
        found = False
        source_y = pred_c[y]
        t_y = pred[y][-1]
        if t_y != 0:
            target_y = pred_c.get(t_y)
            if target_y is None:
                fp += 1
                continue
        else:
            target_y = (None, None)

        for x in truth:
            source_x = truth_c[x]
            t_x = truth[x][-1]
            target_x = truth_c[t_x] if t_x != 0 else (None, None)
            if (
                truth[x][0] == pred[y][0]
                and checkApproxMatch((x, source_x), (y, source_y), typeRequirement=False)
                and checkApproxMatch((t_x, target_x), (t_y, target_y), typeRequirement=False)
            ):
                found = True
                break
        if not found:
            fp += 1

    return tp, fp, fn


def compute_metrics_relation(pred_file, truth_file, ratio=0.9999):
    del ratio
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
            pred_rel = extractRelations(doc, argTypesDocs[idoc])
        except IndexError:
            sys.stderr.write(f"ERROR in doc {idoc}\n")
            continue
        truth_c = extractComponents(truthDocs[idoc], argTypesDocsTruth[idoc])
        truth_rel = extractRelations(truthDocs[idoc], argTypesDocsTruth[idoc])
        tp, fp, fn = computeF1_relations_approx(pred_rel, truth_rel, pred_c, truth_c)
        tps += tp
        fps += fp
        fns += fn

    f1 = 2 * tps * 1.0 / max((2 * tps + fps + fns), 1)
    recall = tps / max((tps + fns), 1)
    precision = tps / max((tps + fps), 1)
    return precision, recall, f1
