import pandas as pd
import os

aug_cols = ["x", "y", "health", "relevant"]

class CSVAugmented:
    def __init__(self, root):
        self.root = root
        self.related = pd.DataFrame(columns=aug_cols)
    
    # data is a list of all the previous data frames
    def jungler(self, point, health, proc_dat):
        all_imgs = []
        for d in proc_dat:
            all_imgs += d.get_frames(point[0], point[1]).tolist()
        
        # print(all_imgs)

        self.related.loc[len(self.related.index)] = [point[0], point[1], health, all_imgs]

    def done(self):
        self.related.to_pickle(os.path.join(self.root, "related.pkl"))