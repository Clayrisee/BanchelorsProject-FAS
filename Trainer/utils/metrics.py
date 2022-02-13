import numpy as np


def APCER(fp, tn):
    return 0 if ((tn+fp) == 0) else float(fp) / float(tn+fp)

def NPCER(tp, fn):
    return 0 if ((fn+tp) == 0 )else float(fn)/ float(fn+tp)

def ACER(apcer, npcer):
    return (apcer + npcer) / 2

def calculate_liveness_metric(y_probs, y_true, threshold=0.5):
    result_metrics=dict()
    # y_probs = 1 - y_probs
    y_pred = np.array(np.greater(y_probs, threshold), dtype=np.float32)
    tp = np.sum(np.logical_and(y_pred, y_true))
    fp = np.sum(np.logical_and(y_pred, np.logical_not(y_true)))
    tn = np.sum(np.logical_and(np.logical_not(y_pred), np.logical_not(y_true)))
    fn = np.sum(np.logical_and(np.logical_not(y_pred), y_true))

    result_metrics['tpr']= 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    result_metrics['fpr']= 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    apcer = APCER(fp, tn)
    npcer = NPCER(tp, fn)
    acer = ACER(apcer, npcer)
    result_metrics['apcer'] = apcer
    result_metrics['npcer'] = npcer
    result_metrics['acer'] = acer
  
    return result_metrics