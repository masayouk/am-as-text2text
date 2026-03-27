"""AAEC CoNLL reader utilities.

Python 3 reimplementation of the original Python 2 AAEC evaluation helper code
from UKPLab/acl2017-neural_end2end_am, released with Eger et al.,
"Neural End-to-End Learning for Computational Argumentation Mining"
(ACL 2017).

Paper: https://aclanthology.org/P17-1002/
GitHub: https://github.com/UKPLab/acl2017-neural_end2end_am
Upstream code license: Apache License 2.0
"""

import random
import sys


def readDocs(fn):
    docs = []
    doc = []
    with open(fn, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line == "":
                if doc != []:
                    docs.append(doc)
                doc = []
            else:
                doc.append(line)
    if doc != []:
        docs.append(doc)
    return docs


def readDocsFine2(fn, field):
    docs = []
    doc = [[]]
    argTypes = []
    atype = []
    lastLabel = None
    with open(fn, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line == "":
                if doc != [[]]:
                    docs.append(doc)
                    argTypes.append(atype)
                doc = [[]]
                atype = []
                lastLabel = None
            else:
                row = line.split("\t")
                label = row[field]
                if label.startswith("B-"):
                    atype.append(label.split(":")[0])
                    if doc[-1] != []:
                        doc.append([])
                elif label.startswith("O") and lastLabel != "O" and lastLabel:
                    atype.append(None)
                    if doc[-1] != []:
                        doc.append([])
                elif label.startswith("O") and lastLabel != "O":
                    atype.append(None)
                doc[-1].append(line)
                lastLabel = label[0] if label else None
    if doc != [[]]:
        docs.append(doc)
        argTypes.append(atype)
    return docs, argTypes


if __name__ == "__main__":  # pragma: no cover
    docs = readDocs(sys.argv[1])
    random.shuffle(docs)
    n = int(sys.argv[2])

    for doc in docs[:n]:
        for line in doc:
            row = line.split("\t")
            row[0] = row[0].split("_")[-1]
            print("\t".join(row))
        print()

    for doc in docs[n:]:
        for line in doc:
            row = line.split("\t")
            row[0] = row[0].split("_")[-1]
            sys.stderr.write("\t".join(row) + "\n")
        sys.stderr.write("\n")
