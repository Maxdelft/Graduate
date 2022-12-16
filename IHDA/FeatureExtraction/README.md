# How to run feature extraction file.

Steps:
1. Guarantee that you have downloaded and stored the data as described in the *README.md* of the folder *SHLData*
2. In the python file generate_fft_features.py specify the variable dir_shl_data which represents the directory of the folder: 'SHLData/2018/'
3. Run the python file: generate_fft_features.py
4. Relax a while, since this will take time. (Probably, this step could be speeded up by switching from numpy to torch).
5. The extracted features are stored in the file: 'SHLData/2018/Features/'. Those files contain the features used for training the models.
