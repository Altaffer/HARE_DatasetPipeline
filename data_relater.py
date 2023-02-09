import pandas as pd
import banners
import csv_relations
import numpy as np

cols = ["image", "points"]

class Relater:
    def __init__(self):
       pass
        
    def clicks_to_image(self, clicks_o, imgs_o):
        c2i_df = pd.DataFrame(columns=cols)
        for l in imgs_o.parsed:
            p = l['pose']
            pts, health = clicks_o.get_pts(p, imgs_o.cone)
            pts = self.convert_to_local(p, pts)
            pxs = clicks_o.project(pts)
            c2i_df.loc[len(c2i_df.index)] = [p[0,3], p[1,3], pxs, health]


    # make points relative to the robot pose 
    def convert_to_local(self, pose, pts):
        points = np.asarray(pts)
        return pts - np.array([pose[0,3], pose[1,3]])
        

    def temporal(self):
        pass