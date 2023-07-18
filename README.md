# Dataset Description
The task is to discriminate between two classes of brain CT scans, one that contains anomalies (label 1) and one that is normal (class 0). Each sample is a grayscale image of 224x224 pixels.

Each example is assigned to one of the two classes. The training set consists of 15,000 labeled examples. The validation set consists of 2,000 labeled examples. The test set consists of another 5,149 examples. The test labels are not provided with the data.

# File descriptions
data.zip - the image samples (one sample per .PNG file)<br>
train_labels.txt - the training labels (one label per row)<br>
validation_labels.txt - the training labels (one label per row)<br>
sample_submission.csv - a sample submission file in the correct format<br>

# Metadata file format
The labels associtated to the training samples are provided in the train_labels_.txt file with the following format:

> **id,class**<br> 
> 000001,0<br>
> ...<br>
> 017000,1<br>

For example, the first row indicates that the data sample file named '000001.png' belongs to class 0 (no bleeding).
