import pickle
import numpy as np
def load_pickle(pickle_file):
    """
    This function loads .pkl format
    Each .pkl contains imporession score for train, validation and test subjects
    """
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def Annotation_to_Numpy(data):
    """
    name of columns for the labels id: each subject, the other columns are characteristics
    interview column is ommited
    """
    characteristics = list(data.keys())
    characteristics.remove('interview')
    Columns = ['id'] + characteristics
    labels = np.array([Columns])
    shape = (len(data['interview']), labels.shape[1])
    subject_id = list(data['interview'].keys())
    labels = np.vstack([labels, np.zeros(shape)])
    labels[1:,0] = subject_id

    for key in Columns[1:]:
        idx = np.argwhere(labels[0, :] == key)
        vals = list(data[key].values())
        labels[1:, idx[0, 0]] = np.array(vals, dtype=np.float32)
    labels_dict = {}
    for label in labels[1:, 0]:
        idx = np.argwhere(labels[:, 0] == label)
        labels_dict[label] = labels[idx[0, 0], 1:]
    labels = labels[1:,:]
    return labels, labels_dict
#%% main function
# annotation_training = load_pickle('annotation_training.pkl')
# annotation_training = Annotation_to_Numpy(annotation_training)
