#import dependencies
import pandas as pd
import numpy as np
import time
import pickle
import config

from pipeline import Pipeline

pipeline = Pipeline()


if __name__ == '__main__':
    
    # load data set
    start_time = time.time()
    #load prosses_table
    log = pd.read_csv(config.Processes_Table)
    #variabel table
    variable_Table = pd.read_csv(config.variable_table)
    #call Preprocessing wraper function
    # pipeline.Preprocessing_orchestrator(log,variable_Table)
    pipeline.scoring_orchestrator()
    pipeline.maintain_inprogress_process(1)
    pipeline.maintain_inprogress_process(2)
    pipeline.maintain_completed_process(1)
    pipeline.maintain_completed_process(2)
    # pipeline.check_comparison()
    print("--- %s seconds ---" % (time.time() - start_time))
    # pipeline.train_test_split()
    # pipeline.save_train_test_data()
    print("Success")