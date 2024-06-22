import numpy as np
import os
import glob
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

def analyze_files(file1, file2):
    # Load the files
    a = np.loadtxt(file1)
    b = np.loadtxt(file2)
    
    # Check if the lengths are equal
    assert len(a) == len(b), "Files have different lengths"
    
    # Initialize counters for each category in the confusion matrix
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    
    # Iterate over the elements of the arrays
    for t, s in zip(a, b):
        if t == 1 and s == 1:
            true_positive += 1
        elif t == 1 and s == 0:
            false_positive += 1
        elif t == 0 and s == 1:
            false_negative += 1
        elif t == 0 and s == 0:
            true_negative += 1
    
    # Return the confusion matrix as a dictionary
    return {
        "True Positive": true_positive,
        "False Positive": false_positive,
        "False Negative": false_negative,
        "True Negative": true_negative
    }

def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def plot_confusion_matrix(confusion_matrix, output_file):
    labels = ["True Positive", "False Positive", "False Negative", "True Negative"]
    values = [confusion_matrix[label] for label in labels]
    
    fig, ax = plt.subplots()
    ax.barh(labels, values, color=['green', 'red', 'red', 'green'])
    ax.set_xlabel('Count')
    ax.set_title('Confusion Matrix Summary')
    
    for i, v in enumerate(values):
        ax.text(v + 1, i, str(v), color='black', va='center')
    
    plt.savefig(output_file)
    plt.close()

def main():
    root = './src/output'
    gt_root = '/home/user/noah/trivial_wall/data/Q2_TW'
    indices = os.listdir(root)
    
    # Initialize summary dictionary
    summary = {
        "True Positive": 0,
        "False Positive": 0,
        "False Negative": 0,
        "True Negative": 0
    }
    
    for index in tqdm(indices):
        pred_list = glob.glob(os.path.join(root, index, '*.txt'))
        for pred_txt_path in pred_list:
            pred_pano_name = pred_txt_path.split('/')[-1].replace('_TW', '')
            gt_txt_path = os.path.join(gt_root, index + '_' + pred_pano_name)
            assert os.path.exists(gt_txt_path), f"{index + '_' + pred_pano_name}: No such a file in GT"
            
            result = analyze_files(pred_txt_path, gt_txt_path)
            
            # Update summary with results
            for key in summary:
                summary[key] += result[key]
    
    # Save summary to JSON file
    json_filename = 'confusion_matrix_summary.json'
    save_json(summary, json_filename)
    
    # Save plot to image file
    plot_filename = 'confusion_matrix_summary.png'
    plot_confusion_matrix(summary, plot_filename)

if __name__ == '__main__':
    main()
