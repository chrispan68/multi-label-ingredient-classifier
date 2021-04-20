import sys, os, io, argparse
import matplotlib.pyplot as plt
import numpy as np

def get_precision_recall_f1(filename):
    with open(filename) as f:
        lines = f.readlines()
        num_examples = []
        precision = []
        recall = []
        f1 = []

        # Loop through lines for each ingredient
        for line in lines[7:360]:
            vals = line.split()[-4:]
            precision.append(float(vals[0]))
            recall.append(float(vals[1]))
            f1.append(float(vals[2]))
            num_examples.append(int(vals[3]))
        
        return np.asarray(num_examples), np.asarray(precision), np.asarray(recall), np.asarray(f1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default='test-0.5-threshold.out')
    args = parser.parse_args()
    args = vars(args)

    filename = args['filename']
    if not os.path.exists('analysis'):
        os.mkdir('analysis')
    
    num_examples, precision, recall, f1 = get_precision_recall_f1(filename)
    # plot precision
    plt.title('Precision as a Function of Number of Examples')
    plt.xlabel('Number of Examples')
    plt.ylabel('Precision')
    plt.xscale('log')
    plt.plot(num_examples, precision, '-o', linestyle='None')

    rho = np.corrcoef(np.log(num_examples), precision)[0][1]
    str_text = 'Pearson correlation: {:.4f}'.format(rho)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    #place in bottom right corner
    plt.text(200, 0.025, str_text, fontsize=10,
        verticalalignment='top', bbox=props)
    
    plt.savefig('analysis/precision.jpg')
    plt.show()
    plt.clf()

    # plot recall
    plt.title('Recall as a Function of Number of Examples')
    plt.xlabel('Number of Examples')
    plt.ylabel('Recall')
    plt.xscale('log')
    plt.plot(num_examples, recall, '-o', linestyle='None')

    rho = np.corrcoef(np.log(num_examples), recall)[0][1]
    str_text = 'Pearson correlation: {:.4f}'.format(rho)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    #place in bottom right corner
    plt.text(200, 0.025, str_text, fontsize=10,
        verticalalignment='top', bbox=props)

    plt.savefig('analysis/recall.jpg')
    plt.show()
    plt.clf()

    # plot f1-score
    plt.title('F1-Score as a Function of Number of Examples')
    plt.xlabel('Number of Examples')
    plt.ylabel('F1-Score')
    plt.xscale('log')
    plt.plot(num_examples, f1, '-o', linestyle='None')

    rho = np.corrcoef(np.log(num_examples), f1)[0][1]
    str_text = 'Pearson correlation: {:.4f}'.format(rho)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    #place in bottom right corner
    plt.text(200, 0.025, str_text, fontsize=10,
        verticalalignment='top', bbox=props)

    plt.savefig('analysis/f1-score.jpg')
    plt.show()
    plt.clf()
