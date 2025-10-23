# DT2470-HT25-Final-Project
Add some description about the project later.

## Environment

This project is built under python 3.10.18.
Please install python dependencies by the following command:

```bash
pip install -r requirements.txt
```

This project uses [covers80](http://labrosa.ee.columbia.edu/projects/coversongs/covers80/) [1] dataset.
Please download and extract the dataset.

```bash
wget http://labrosa.ee.columbia.edu/projects/coversongs/covers80/covers80.tgz
tar -xvzf covers80.tgz
```

ffmpeg packet is needed in order to run this project. Install with:

```bash
sudo apt update
sudo apt install ffmpeg=7:6.1.1-3ubuntu5
```

## File Structure

- `algorithms.py`: File for matching algorithms.
- `utils.py`: Functions for loading data and other useful functions.
- `main.py`: File for main logic.

Other files are for experimental purposes and might not be of use.

## Result

Result of matched scores of the first 25 songs are as follows.
The score are standardized per row.
Green dots shows correct matches, while red dots shows incorrect ones.

![Score matrix for first 25 songs](/score_matrix/score_matrix_25.png)

Result of matched scores of all songs in `covers80` dataset:

![Score matrix for all songs](/score_matrix/score_matrix_80.png)


## Reference

[1] D. P. W. Ellis (2007). The "covers80" cover song data set.
    Web resource, available: http://labrosa.ee.columbia.edu/projects/coversongs/covers80/. 

## License
Add a license (or not) later.
