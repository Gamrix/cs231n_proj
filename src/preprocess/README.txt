Get dataset from the following address and put it in the dataset folder.

# Download data, about 1.2GB
wget --no-check-certificate "https://onedrive.live.com/download?cid=A1CBC646F973B148&resid=A1CBC646F973B148%2182122&authkey=AHunCElXA7huJPQ"

# Rename it 
mv download\?cid\=A1CBC646F973B148\&resid\=A1CBC646F973B148%2182122\&authkey\=AHunCElXA7huJPQ kitti_simple.zip

# Run this from the src/ folder - mv kitti_simple if necessary
unzip kitti_sample.zip -d preprocess/prep_res
