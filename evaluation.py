import numpy as np
import json, glob, os
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr

data_dir = '/Users/sharon/data/EE148/affordance/Data/'

median_pinch = 62.
median_clench = 75.
median_poke = 48.
median_palm = 46.

GRASP_ID = 5
WGRASP_ID = 9

def segmentation_IoU_helper(segment_label, segment_pred):
    prehensile_label = np.logical_or(segment_label == GRASP_ID, segment_label == WGRASP_ID)
    prehensile_pred = segment_pred == 1
    intersection = np.sum(np.logical_and(prehensile_label, prehensile_pred))
    label_area = np.sum(prehensile_label)
    pred_area = np.sum(prehensile_pred)
    IoU = intersection / (label_area + pred_area - intersection)
    return IoU

def segmentation_IoU_from_np_batches(segment_labels, segment_preds):
    '''
    Input
    - segment_labels: batch_size x image_width x image_height np array, 5 or 9 are the graspable pixels, 0 o.w.
    - segment_preds: batch_size x image_width x image_height np array, 1 are the graspable pixels, 0 o.w.

    Return
    - average IoU score
    '''
    IoU_list = []
    for i in range(segment_labels.shape[0]):
        segement_label = segment_labels[i]
        segement_pred = segment_preds[i]
        IoU = segmentation_IoU_helper(segment_label, segment_pred)
        IoU_list.append(IoU)

    average_IoU = np.average(IoU_list)
    print("==Segmentation==")
    print("IoU: ", average_IoU)
    return average_IoU


def segmentation_IoU_from_files(segment_labels_path, segment_preds_path):
    '''
    Input
    - segment_labels_path: path to Daniel's affordance_labels numpy files. (e.g., "./IIT_AFF_processed/affordance_labels/")
    - segment_preds_path: path to the predicted affordance segmentation numpy files. (e.g., "./IIT_AFF_processed/affordance_preds/")

    Return
    - average IoU score
    '''
    # Build a list of numpy file names
    segment_files = [file_path.split(os.sep)[-1] for file_path in glob.glob(os.path.join(segment_labels_path, '*.npy'))]
    IoU_list = []
    for segment_file in segment_files:
        segment_label = np.load(open(os.path.join(segment_labels_path, segment_file), 'rb'))
        segment_pred = np.load(open(os.path.join(segment_preds_path, segment_file), 'rb'))
        IoU = segmentation_IoU_helper(segment_label, segment_pred)
        IoU_list.append(IoU)

    average_IoU = np.average(IoU_list)
    print("==Segmentation==")
    print("IoU: ", average_IoU)
    return average_IoU


def score_evaluation_from_np_batches(score_label, score_pred):
    '''
    Input
    - score_label: batch_size x 4 np array of ground truth affordance scores
    - score_pred: batch_size x 4 np array of predicted affordance scores

    First to fourth columns are pinch, clench, poke and palm scores each

    Return
    - averge MSE, average pearson correlation, average accuracy
    '''
    batch_size = len(score_label)

    mse = ((score_label - score_pred)**2).mean(axis=0)

    pinch_mse = mse[0]
    clench_mse = mse[1]
    poke_mse = mse[2]
    palm_mse = mse[3]

    pinch_corr = pearsonr(score_label[:, 0], score_pred[:, 0])
    clench_corr = pearsonr(score_label[:, 1], score_pred[:, 1])
    poke_corr = pearsonr(score_label[:, 2], score_pred[:, 2])
    palm_corr = pearsonr(score_label[:, 3], score_pred[:, 3])

    binarized_label = np.zeros((batch_size, 4))
    binarized_pred = np.zeros((batch_size, 4))

    binarized_label[:, 0] = score_label[:, 0] > median_pinch
    binarized_label[:, 1] = score_label[:, 1] > median_clench
    binarized_label[:, 2] = score_label[:, 2] > median_poke
    binarized_label[:, 3] = score_label[:, 3] > median_palm

    binarized_pred[:, 0] = score_pred[:, 0] > median_pinch
    binarized_pred[:, 1] = score_pred[:, 1] > median_clench
    binarized_pred[:, 2] = score_pred[:, 2] > median_poke
    binarized_pred[:, 3] = score_pred[:, 3] > median_palm

    pinch_acc = accuracy_score(binarized_label[:, 0], binarized_pred[:, 0])
    clench_acc = accuracy_score(binarized_label[:, 1], binarized_pred[:, 1])
    poke_acc = accuracy_score(binarized_label[:, 2], binarized_pred[:, 2])
    palm_acc = accuracy_score(binarized_label[:, 3], binarized_pred[:, 3])

    print("=====MSE=====")
    print("Pinch: ", pinch_mse, "Clench: ", clench_mse, "Poke: ", poke_mse, "Palm: ", palm_mse)
    mean_mse = np.mean(mse)
    print("Average: ", mean_mse)

    print("=====Corr=====")
    print("Pinch: ", pinch_corr, "Clench: ", clench_corr, "Poke: ", poke_corr, "Palm: ", palm_corr)
    mean_corr = np.mean([pinch_corr[0], clench_corr[0], poke_corr[0], palm_corr[0]])
    print("Average: ", mean_corr)

    print("=====Acc=====")
    print("Pinch: ", pinch_acc, "Clench: ", clench_acc, "Poke: ", poke_acc, "Palm: ", palm_acc)
    mean_acc = np.mean([pinch_acc, clench_acc, poke_acc, palm_acc])
    print("Average: ", mean_acc)

    return mean_mse, mean_corr, mean_acc


def score_evaluation_from_files(aff_score_labels_path, aff_score_preds_path):
    '''
    Input
    - aff_score_labels_path: path to the ground truth affordance score json file. The json file should have the format of {<image name1>: {'pinch': 44.2, 'clench': 83, 'poke': 38.1, 'palm': 23.9}, <image name2>: {'pinch': 14.2, 'clench': 41.1, 'poke': 28.0, 'palm': 63.3}, ...}
    - aff_score_preds_path: path to the predicted affordance score json file. The format should be identical with the ground truth json file.

    Return
    averge MSE, average pearson correlation, average accuracy
    '''
    labels = json.load(open(aff_score_labels_path, 'r'))
    preds = json.load(open(aff_score_preds_path, 'r'))

    l = len(labels)

    score_label = np.zeros((l, 4))
    score_pred = np.zeros((l, 4))

    for i, (file_name, label_scores) in enumerate(labels.items()):
        pred_scores = preds[file_name]

        score_label[i, 0] = label_scores['pinch']
        score_label[i, 1] = label_scores['clench']
        score_label[i, 2] = label_scores['poke']
        score_label[i, 3] = label_scores['palm']

        score_pred[i, 0] = pred_scores['pinch']
        score_pred[i, 1] = pred_scores['clench']
        score_pred[i, 2] = pred_scores['poke']
        score_pred[i, 3] = pred_scores['palm']


    mean_mse, mean_corr, mean_acc = score_evaluation_from_np_batches(score_label, score_pred)

    return mean_mse, mean_corr, mean_acc



if __name__ == "__main__":
    # This part is for testing the implemented methods.
    AFF_SCORE_LABEL_PATH = '%s/score_labels.json' % data_dir # Path to the json file
    AFF_SCORE_PRED_PATH = '%s/score_labels.json' % data_dir
    SEGMENT_LABEL_PATH = '%s/IIT_AFF_processed/affordances_labels/' % data_dir # Path to the npy files
    SEGMENT_PRED_PATH = '%s/IIT_AFF_processed/affordances_labels/' % data_dir

    score_evaluation_from_files(AFF_SCORE_LABEL_PATH, AFF_SCORE_PRED_PATH)
    segmentation_IoU_from_files(SEGMENT_LABEL_PATH, SEGMENT_PRED_PATH)
