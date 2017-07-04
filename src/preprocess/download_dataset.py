from __future__ import print_function

import glob
import logging
import os
import shutil
import zipfile
from multiprocessing import pool
from urllib.request import urlretrieve

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d %I:%M:%S', level=logging.INFO)

dst = "J:/kitti/dataset"
src = "http://kitti.is.tue.mpg.de/kitti/raw_data/"


def download_data():
    os.makedirs(dst + "/zips", exist_ok=True)  # yeah, shouldn't be using this...
    os.chdir(dst)

    # limit concurrent writters to external disk
    # disk_sema = Semaphore(2)

    with pool.Pool(6) as p:
        # p.map(download_file, zip(files.split(), itertools.repeat(disk_sema)))
        p.map(download_file, files.split())


def download_file(file):
    try:
        if not len(file): return

        if file[-3:] == "zip":
            shortname = file
            longname = file
        else:
            shortname = file + "_sync.zip"
            longname = file + "/" + shortname

        out_folder = shortname[:-4]
        if not os.path.exists("zips/" + shortname):
            logging.info("Downloading " + shortname)
            urlretrieve(src + longname, "zips/" + shortname)

        os.makedirs(out_folder, exist_ok=True)  # yeah, shouldn't be using this...

        if os.path.exists(out_folder + "/finished.txt"): return

        zip_dest = "./{}/extract".format(out_folder)
        try:
            with zipfile.ZipFile("zips/" + shortname, 'r') as zip:
                logging.info("Extracting Files for " + shortname)
                all_files = zip.namelist()
                image_files = [f for f in all_files if "image_02" in f]

                for f in image_files:
                    zip.extract(f, path=zip_dest)
        except zipfile.BadZipFile as e:
            logging.warning("Deleting bad zip file " + shortname)
            os.remove("zips/" + shortname)
            return

        # now move all the files over
        for pic in glob.iglob(zip_dest + "/**/*.png", recursive=True):
            shutil.move(pic, "./{}/{}".format(out_folder, os.path.basename(pic)))

        # remove the extracted stuff
        shutil.rmtree(zip_dest)

        with open(out_folder + "/finished.txt", "w") as f:
            f.write("Done")

    except Exception as e:
        import traceback

        logging.error("Failure with {}\n".format(file) + traceback.format_exc())


files = """
2011_09_26_calib.zip
2011_09_26_drive_0001
2011_09_26_drive_0002
2011_09_26_drive_0005
2011_09_26_drive_0009
2011_09_26_drive_0011
2011_09_26_drive_0013
2011_09_26_drive_0014
2011_09_26_drive_0015
2011_09_26_drive_0017
2011_09_26_drive_0018
2011_09_26_drive_0019
2011_09_26_drive_0020
2011_09_26_drive_0022
2011_09_26_drive_0023
2011_09_26_drive_0027
2011_09_26_drive_0028
2011_09_26_drive_0029
2011_09_26_drive_0032
2011_09_26_drive_0035
2011_09_26_drive_0036
2011_09_26_drive_0039
2011_09_26_drive_0046
2011_09_26_drive_0048
2011_09_26_drive_0051
2011_09_26_drive_0052
2011_09_26_drive_0056
2011_09_26_drive_0057
2011_09_26_drive_0059
2011_09_26_drive_0060
2011_09_26_drive_0061
2011_09_26_drive_0064
2011_09_26_drive_0070
2011_09_26_drive_0079
2011_09_26_drive_0084
2011_09_26_drive_0086
2011_09_26_drive_0087
2011_09_26_drive_0091
2011_09_26_drive_0093
2011_09_26_drive_0095
2011_09_26_drive_0096
2011_09_26_drive_0101
2011_09_26_drive_0104
2011_09_26_drive_0106
2011_09_26_drive_0113
2011_09_26_drive_0117
2011_09_26_drive_0119
2011_09_28_calib.zip
2011_09_28_drive_0001
2011_09_28_drive_0002
2011_09_28_drive_0016
2011_09_28_drive_0021
2011_09_28_drive_0034
2011_09_28_drive_0035
2011_09_28_drive_0037
2011_09_28_drive_0038
2011_09_28_drive_0039
2011_09_28_drive_0043
2011_09_28_drive_0045
2011_09_28_drive_0047
2011_09_28_drive_0053
2011_09_28_drive_0054
2011_09_28_drive_0057
2011_09_28_drive_0065
2011_09_28_drive_0066
2011_09_28_drive_0068
2011_09_28_drive_0070
2011_09_28_drive_0071
2011_09_28_drive_0075
2011_09_28_drive_0077
2011_09_28_drive_0078
2011_09_28_drive_0080
2011_09_28_drive_0082
2011_09_28_drive_0086
2011_09_28_drive_0087
2011_09_28_drive_0089
2011_09_28_drive_0090
2011_09_28_drive_0094
2011_09_28_drive_0095
2011_09_28_drive_0096
2011_09_28_drive_0098
2011_09_28_drive_0100
2011_09_28_drive_0102
2011_09_28_drive_0103
2011_09_28_drive_0104
2011_09_28_drive_0106
2011_09_28_drive_0108
2011_09_28_drive_0110
2011_09_28_drive_0113
2011_09_28_drive_0117
2011_09_28_drive_0119
2011_09_28_drive_0121
2011_09_28_drive_0122
2011_09_28_drive_0125
2011_09_28_drive_0126
2011_09_28_drive_0128
2011_09_28_drive_0132
2011_09_28_drive_0134
2011_09_28_drive_0135
2011_09_28_drive_0136
2011_09_28_drive_0138
2011_09_28_drive_0141
2011_09_28_drive_0143
2011_09_28_drive_0145
2011_09_28_drive_0146
2011_09_28_drive_0149
2011_09_28_drive_0153
2011_09_28_drive_0154
2011_09_28_drive_0155
2011_09_28_drive_0156
2011_09_28_drive_0160
2011_09_28_drive_0161
2011_09_28_drive_0162
2011_09_28_drive_0165
2011_09_28_drive_0166
2011_09_28_drive_0167
2011_09_28_drive_0168
2011_09_28_drive_0171
2011_09_28_drive_0174
2011_09_28_drive_0177
2011_09_28_drive_0179
2011_09_28_drive_0183
2011_09_28_drive_0184
2011_09_28_drive_0185
2011_09_28_drive_0186
2011_09_28_drive_0187
2011_09_28_drive_0191
2011_09_28_drive_0192
2011_09_28_drive_0195
2011_09_28_drive_0198
2011_09_28_drive_0199
2011_09_28_drive_0201
2011_09_28_drive_0204
2011_09_28_drive_0205
2011_09_28_drive_0208
2011_09_28_drive_0209
2011_09_28_drive_0214
2011_09_28_drive_0216
2011_09_28_drive_0220
2011_09_28_drive_0222
2011_09_28_drive_0225
2011_09_29_calib.zip
2011_09_29_drive_0004
2011_09_29_drive_0026
2011_09_29_drive_0071
2011_09_29_drive_0108
2011_09_30_calib.zip
2011_09_30_drive_0016
2011_09_30_drive_0018
2011_09_30_drive_0020
2011_09_30_drive_0027
2011_09_30_drive_0028
2011_09_30_drive_0033
2011_09_30_drive_0034
2011_09_30_drive_0072
2011_10_03_calib.zip
2011_10_03_drive_0027
2011_10_03_drive_0034
2011_10_03_drive_0042
2011_10_03_drive_0047
2011_10_03_drive_0058
"""

if __name__ == '__main__':
    download_data()
