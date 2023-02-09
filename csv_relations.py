import pandas as pd
import os
import csv
# import pymap3d as pm
import utm

aug_cols = ["x", "y", "health"]

class CSVAugmented:
    def __init__(self, file):
        self.csv_file = file
        self.click_data = pd.DataFrame(columns=aug_cols)
    
    # data is a list of all the previous data frames
    def jungler(self, point, radius):
        pass

    def done(self):
        self.related.to_pickle(os.path.join(self.root, "related.pkl"))

    def csv_read(self):
        with open(self.csv_file) as clicks:
            reader = csv.reader(clicks)
            for line in reader:
                if origin == None:
                    origin = [float(line[0]), float(line[1]), float(line[2])]
                # breakdown line
                u = utm.from_latlon(float(line[0]), float(line[1]))
                # u = pm.geodetic2ned(float(line[0]), float(line[1]), float(line[2]), origin[0], origin[1], origin[2])
                # print(u)
                health = int(line[-1])
                
                self.click_data.loc[len(self.click_data.index)] = [u[0], u[1], health]

    def get_pts(self, pose, cone):
        pass
        # query for points
