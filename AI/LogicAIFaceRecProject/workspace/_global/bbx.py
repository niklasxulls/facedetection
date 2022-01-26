import json
class Bbx:
    def __init__(self, path, xmin, ymin, xmax, ymax):
        self.path = path
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)