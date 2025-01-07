from utils import TrainOptions
from utils import TrainOptions_meta
from train import Trainer_cliff, Trainer_hmr, Trainer_mutilROI, Trainer_metaHMR

if __name__ == '__main__':
    # options = TrainOptions().parse_args()
    options = TrainOptions_meta().parse_args()
    print(f'==================using model {options.model_name}=====================')
    if options.model_name == 'cliff':
        trainer = Trainer_cliff(options)
    elif options.model_name == 'hmr':
        trainer = Trainer_hmr(options)
    elif options.model_name == 'mutilROI':
        trainer = Trainer_mutilROI(options)
    elif options.model_name == 'metaHMR':
        trainer = Trainer_metaHMR(options)
    trainer.train()
