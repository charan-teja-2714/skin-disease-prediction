from collections import defaultdict

def confusion_stats(preds, labels):
    stats = defaultdict(int)
    for p, y in zip(preds, labels):
        if p == 1 and y == 1:
            stats["TP"] += 1
        elif p == 0 and y == 0:
            stats["TN"] += 1
        elif p == 1 and y == 0:
            stats["FP"] += 1
        elif p == 0 and y == 1:
            stats["FN"] += 1
    return stats

def false_negative_rate(stats):
    fn = stats["FN"]
    tp = stats["TP"]
    return fn / (fn + tp + 1e-6)

def precision(stats):
    tp = stats["TP"]
    fp = stats["FP"]
    return tp / (tp + fp + 1e-6)

def recall(stats):
    tp = stats["TP"]
    fn = stats["FN"]
    return tp / (tp + fn + 1e-6)
