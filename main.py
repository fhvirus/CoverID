from utils import load_covers80, match_all_songs
from algorithms import dummy_compare, compare

DATADIR = './coversongs/covers32k/'

if __name__ == '__main__':
    name_list, ver_A, ver_B = load_covers80(DATADIR)
    truth_list, matched_list = match_all_songs(ver_A, ver_B, compare)

    precision = sum([ a == b for a, b in zip(truth_list, matched_list) ]) / len(truth_list)
    print(f'Precision: {precision}')

    # confusion_matrix = sklearn.metrics.classification_report(truth_list, matched_list)
    # print(confusion_matrix)
