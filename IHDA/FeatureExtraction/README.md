# How to run feature extraction file.

Steps:
1. Guarantee that you have downloaded and stored the data as described in SHLData/README.md
2. In generate_fft_features.py specify the variable dir_shl_data which represents the directory of the folder: 'SHLData/2018/'
3. Run the generate_fft_features.py
4. Relax a while. This takes a while since it uses numpy instead of torch. Rewriting to torch could speed up this process.
5. Extracted features are stored: 'SHLData/2018/Features/'
