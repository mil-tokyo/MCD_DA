
class BaseDataLoader():
    def __init__(self):
        pass
    
    def initialize(self,batch_size):
        self.batch_size = batch_size
        self.serial_batches = 0
        self.nThreads = 2
        self.max_dataset_size=float("inf")
        pass

    def load_data():
        return None

        
        
