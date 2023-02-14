import pandas as pd
import os
import csv
# import pymap3d as pm
import utm
import pickle

aug_cols = ["x", "y", "health"]

class CSVAugmented:
    def __init__(self, file, root):
        self.csv_file = file
        self.root = root
        self.click_data = pd.DataFrame(columns=aug_cols)
    

    # data is a list of all the previous data frames
    def jungler(self, point, radius):
        pass


    def done(self):  
        self.click_data.to_pickle(os.path.join(self.root, "related.pkl"))
        # with open(os.path.join('data/test_dir/', "related.pkl"), 'wb') as f:
        #     pickle.dump(self.__dict__, f, protocol=2)


    def uptake(self):
        self.parsed = pd.read_pickle(os.path.join(self.root, "related.pkl"))


    def csv_read(self):
        origin = None
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

    """ get the clicks within the FOV """
    def get_pts(self, pose, cone):
        got_points = self.click_data.loc[(self.click_data['x'] > cone[0] + pose[0,3] ) & 
                            (self.click_data['x'] < cone[1] + pose[0,3] ) & 
                            (self.click_data['y'] > cone[2] + pose[1,3] ) & 
                            (self.click_data['y'] < cone[3] + pose[1,3] )]
        return got_points
        

def done(fname, obj):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f, protocol=2)