import pickle
import numpy as np
def Annotation_to_Numpy(data):
    # name of columns for the labels
    # id: each subject, the other columns are characteristics
    # interview column is ommited
    characteristics = list(data.keys())
    characteristics.remove('interview')
    Columns = ['id'] + characteristics
    labels = np.array([Columns])
    shape = (len(data['interview']), labels.shape[1])
    subject_id = list(data['interview'].keys())
    labels = np.vstack([labels, np.zeros(shape)])
    labels[1:,0] = subject_id
    
    for key in Columns[1:]:
        idx = np.argwhere(labels[0,:]==key)
        vals = list(data[key].values())
        labels[1:,idx[0,0]] = np.array(vals,dtype='float64')
    return labels
#if __name__=='__main__':
#    main()