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


## File Structure

- `algorithms.py`: File for matching algorithms.
- `utils.py`: Functions for loading data and other useful functions.
- `main.py`: File for main logic.

To add a algorithm, place it in algorithms and change the compare function that `main.py` uses.


## Reference

[1] D. P. W. Ellis (2007). The "covers80" cover song data set.
    Web resource, available: http://labrosa.ee.columbia.edu/projects/coversongs/covers80/. 

## License
Add a license (or not) later.
