from model_hLSTMat.solver import Solver
from model_hLSTMat.model import Model
from data_engine import DataEngine
import argparse


def main(dataset, model_name, num_gpus):
    # load train dataset
    # data = load_coco_data(data_path='./data', split='train')
    # word_to_idx = data['word_to_idx']
    # load val dataset to print out bleu scores every epoch
    # val_data = load_coco_data(data_path='./data', split='val')
    engine = DataEngine()
    if dataset == 'msvd':
        data = engine.msvd()
    elif dataset == 'msrvtt':
        data = engine.msr_vtt()
    # data, val_data, test_data = engine.get_data()
    model = Model(data.vocab.word2idx, dim_feature=[28, 2048], dim_embed=512,
                  dim_hidden=1024, n_time_step=30)

    solver = Solver(model, data, n_epochs=100, batch_size=64, update_rule='adam',
                    learning_rate=0.0001, print_every=25, save_every=10, image_path='./image/',
                    pretrained_model=None, model_path='model/%s/%s' % (dataset, model_name),
                    test_model='model/lstm/model-10',
                    print_bleu=True, log_path='log/%s/%s' % (dataset, model_name), dim_feature=[28, 2048],
                    num_gpus=num_gpus)

    solver.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='msvd')
    parser.add_argument('--model', type=str, default='hLSTMat')
    args = parser.parse_args()
    main(args.dataset, args.model, args.num_gpus)
