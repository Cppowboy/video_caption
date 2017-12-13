from model_hLSTMat.solver import Solver
from model_hLSTMat.model import Model
from data_engine import DataEngine


def main():
    # load train dataset
    # data = load_coco_data(data_path='./data', split='train')
    # word_to_idx = data['word_to_idx']
    # load val dataset to print out bleu scores every epoch
    # val_data = load_coco_data(data_path='./data', split='val')
    engine = DataEngine()
    data = engine.msvd()
    # data, val_data, test_data = engine.get_data()
    model = Model(data.vocab.word2idx, dim_feature=[28, 2048], dim_embed=512,
                  dim_hidden=1024, n_time_step=30)

    solver = Solver(model, data, n_epochs=100, batch_size=64, update_rule='adam',
                    learning_rate=0.0001, print_every=25, save_every=100, image_path='./image/',
                    pretrained_model=None, model_path='model/lstm/', test_model='model/lstm/model-100',
                    print_bleu=True, log_path='log/')

    solver.test(split='test')


if __name__ == "__main__":
    main()
