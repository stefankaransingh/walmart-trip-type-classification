import numpy as np
import matplotlib.pyplot as plt
import operator


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn.metrics import precision_recall_fscore_support


def plot_metric_by_class(y_true,y_pred,labels,ascending=False,metric_to_track='accuracy',save_as=None):
    metric = {}
    overall_score  = 0
    if metric_to_track == 'accuracy':
        overall_score = accuracy_score(y_true,y_pred)
        for label in labels:
            indices = [i for i, x in enumerate(y_true) if x == label]
            metric.update({label:accuracy_score([y_true[i] for i in indices],[y_pred[i] for i in indices])})

    else:
        precision,recall,f1,support = precision_recall_fscore_support(y_true,y_pred,1.0,labels)

        for index,label in enumerate(labels):
            if metric_to_track == 'precision':
                overall_score = precision_score(y_true,y_pred)
                metric.update({label:precision[index]})
            elif metric_to_track == 'recall':
                overall_score = recall_score(y_true,y_pred)
                metric.update({label:recall[index]})
            elif metric_to_track == 'f1':
                overall_score = f1_score(y_true,y_pred,average='weighted')
                metric.update({label:f1[index]})
            else:
                raise ValueError('Please enter a suitable metric to track.')

    metric= sorted(metric.items(), key=operator.itemgetter(1))

    if not ascending:
        metric.reverse()

    scores = list([x[1] for x in metric])
    mean_score = np.array(scores).mean()

    classes = list([x[0] for x in metric])
    y_pos = np.arange(len(classes))


    plt.bar(y_pos, scores, align='center')
    plt.xticks(y_pos, classes)
    plt.ylabel('Metric: {0} | Avg {0} : {1:.3f}'.format(metric_to_track.upper(),mean_score))
    plt.title('{0} by Class | Overall {0} Score: {1}'.format(metric_to_track.upper(),overall_score))

    plt.axhline(mean_score,c='r',ls='--')

    if save_as is not None:
        plt.savefig(save_as)

    plt.show()
