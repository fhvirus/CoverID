# CoverID: A Version Identification Algorithm

CoverID is a version identification system of covers80 dataset [1] based on chroma feature and Smith-Waterman algorithm.

This is a final project of Music Informatics (DT2470 HT25) course at KTH.

## Environment

This project is built under python 3.10.18.
Please install python dependencies by the following command:

```bash
pip install -r requirements.txt
```

This project uses [covers80](http://labrosa.ee.columbia.edu/projects/coversongs/covers80/) [1] dataset. Please download and extract the dataset.

```bash
wget http://labrosa.ee.columbia.edu/projects/coversongs/covers80/covers80.tgz
tar -xvzf covers80.tgz
```

ffmpeg packet is needed in order to run this project. Install with:

```bash
sudo apt update
sudo apt install ffmpeg=7:6.1.1-3ubuntu5
```

## Execution

After setting up the environment and dataset, simply run:

```
python main.py
```

## File Structure

- `algorithms.py`: File for matching algorithms.
- `utils.py`: Functions for loading data and other useful functions.
- `main.py`: File for main logic.
- `figures/`: Figures for report.
- `score_matrix/`: Code and visualization of the matched score of each song.

Other files are for experimental purposes and might not be of use.

## Result

Result of matched scores of the first 25 songs are as follows.
The score are standardized per row.
Green dots shows correct matches, while red dots shows incorrect ones.

![Score matrix for first 25 songs](/score_matrix/score_matrix_25.png)

Result of matched scores of all songs in `covers80` dataset:

![Score matrix for all songs](/score_matrix/score_matrix_80.png)

The accurarcy is:

```
Top-1 Accuracy: 0.533
Top-3 Accuracy: 0.549
Top-5 Accuracy: 0.557
```

Top-k accuracy is calculated as sum of weight for each match divided by the number of songs,
where weight is $1/x$ if the correct match is within the first $k$ largest score and has rank $x$,
and 0 otherwise.


## Reference

[1] D. P. W. Ellis (2007). The "covers80" cover song data set.
    Web resource, available: http://labrosa.ee.columbia.edu/projects/coversongs/covers80/. 
