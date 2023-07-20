from typing import Optional
from . import TaskDataset

class STRLRobotics(TaskDataset):
    def __init__(self, path_to_datset: Optional[str] = None):
        self.path_to_dataset = path_to_datset
        super().__init__()
        
    def get_data(self):
        




        
        
if __name__ == '__main__':
    dataset = STRLRobotics()
    print()
    