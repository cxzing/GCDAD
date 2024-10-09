
import time
from utils.utils import *
from model.GCDAD import GCDAD
from data_factory.data_loader import get_loader_segment
from metrics.metrics import *
import warnings

warnings.filterwarnings('ignore')


def my_kl_loss(p, q):  # 128 1 100 100
    res = p * (torch.log(p + 0.001) - torch.log(q + 0.001))
    return torch.sum(res, dim=-1)  # 128 1 100->128 100


def my_kl_loss_1(p, q):  # 128 1 100 100
    res = p * (torch.log(p + 0.001) - torch.log(q + 0.001))
    return torch.mean(torch.sum(res, dim=1), dim=-1)  # 128 1 100->128 100


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.index, 'dataset/' + self.data_path, batch_size=self.batch_size,
                                               win_size=self.win_size, mode='train', dataset=self.dataset, )
        self.vali_loader = get_loader_segment(self.index, 'dataset/' + self.data_path, batch_size=self.batch_size,
                                              win_size=self.win_size, mode='val', dataset=self.dataset)
        self.test_loader = get_loader_segment(self.index, 'dataset/' + self.data_path, batch_size=self.batch_size,
                                              win_size=self.win_size, mode='test', dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.index, 'dataset/' + self.data_path, batch_size=self.batch_size,
                                              win_size=self.win_size, mode='thre', dataset=self.dataset)
        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def build_model(self):
        self.model = GCDAD(win_size=self.win_size, d_model=self.d_model,  patch_size=self.patch_size,channel=self.input_c,space_num=self.space_num)
        if torch.cuda.is_available():
            self.model.cuda()
        # total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # print("params_num",total_params)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)


    def train(self):

        time_now = time.time()
        train_steps = len(self.train_loader)  # 3866
        # print("r",self.r)
        for epoch in range(self.num_epochs):
            iter_count = 0

            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)  # (128,100,51)
                series, prior, series_1, prior_1 = self.model(input)
                series_loss = 0.0
                prior_loss = 0.0
                series_loss_1 = 0.0
                prior_loss_1 = 0.0
                for u in range(len(prior)):
                    series_loss += (torch.mean(my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1,
                                                                                                   self.d_model)).detach())) + torch.mean(
                        my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1,
                                                                                                           self.d_model)).detach(),
                                   series[u])))
                    prior_loss += (torch.mean(my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1,
                                                                                                self.d_model)),
                        series[u].detach())) + torch.mean(
                        my_kl_loss(series[u].detach(), (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1,
                                                                                                       self.d_model)))))
                    series_loss_1 += (torch.mean(my_kl_loss_1(series_1[u], (
                            prior_1[u] / torch.unsqueeze(torch.sum(prior_1[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.d_model)).detach())) + torch.mean(
                        my_kl_loss_1(
                            (prior_1[u] / torch.unsqueeze(torch.sum(prior_1[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                        self.d_model)).detach(),
                            series_1[u])))
                    prior_loss_1 += (torch.mean(my_kl_loss_1(
                        (prior_1[u] / torch.unsqueeze(torch.sum(prior_1[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.d_model)),
                        series_1[u].detach())) + torch.mean(
                        my_kl_loss_1(series_1[u].detach(), (
                                prior_1[u] / torch.unsqueeze(torch.sum(prior_1[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                           self.d_model)))))

                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)
                series_loss_1 = series_loss_1 / len(prior)
                prior_loss_1 = prior_loss_1 / len(prior)
                loss = prior_loss - series_loss
                loss = loss * self.r + (1 - self.r) * (prior_loss_1 - series_loss_1)
                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                loss.backward()
                self.optimizer.step()
            print(
                "Epoch: {0}, Cost time: {1:.3f}s ".format(
                    epoch + 1, time.time() - epoch_time))
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

    def test(self):
        self.model.eval()
        # print("r", self.r)
        # (1) stastic on the train set
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.train_loader):
            input = input_data.float().to(self.device)
            series, prior, series_1, prior_1 = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            series_loss_1 = 0.0
            prior_loss_1 = 0.0
            for u in range(len(prior)):
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1,
                                                                                                   self.d_model)).detach())
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1,
                                                                                                self.d_model)),
                        series[u].detach())

                    series_loss_1 += my_kl_loss_1(series_1[u], (
                            prior_1[u] / torch.unsqueeze(torch.sum(prior_1[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.d_model)).detach())
                    prior_loss_1 += my_kl_loss_1(
                        (prior_1[u] / torch.unsqueeze(torch.sum(prior_1[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.d_model)),
                        series_1[u].detach())

            metric = torch.softmax(
                (-series_loss - prior_loss) * self.r + (1 - self.r) * (-series_loss_1 - prior_loss_1), dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            series, prior, series_1, prior_1 = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            series_loss_1 = 0.0
            prior_loss_1 = 0.0
            for u in range(len(prior)):
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1,
                                                                                                   self.d_model)).detach())
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1,
                                                                                                self.d_model)),
                        series[u].detach())

                    series_loss_1 += my_kl_loss_1(series_1[u], (
                            prior_1[u] / torch.unsqueeze(torch.sum(prior_1[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.d_model)).detach())
                    prior_loss_1 += my_kl_loss_1(
                        (prior_1[u] / torch.unsqueeze(torch.sum(prior_1[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.d_model)),
                        series_1[u].detach())

            metric = torch.softmax(
                (-series_loss - prior_loss) * self.r + (1 - self.r) * (-series_loss_1 - prior_loss_1), dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print("anormly_ratio",self.anormly_ratio)
        print("Threshold :", thresh)

        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            series, prior, series_1, prior_1 = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            series_loss_1 = 0.0
            prior_loss_1 = 0.0
            for u in range(len(prior)):
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1,
                                                                                                   self.d_model)).detach())
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1,
                                                                                                self.d_model)),
                        series[u].detach())

                    series_loss_1 += my_kl_loss_1(series_1[u], (
                            prior_1[u] / torch.unsqueeze(torch.sum(prior_1[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.d_model)).detach())
                    prior_loss_1 += my_kl_loss_1(
                        (prior_1[u] / torch.unsqueeze(torch.sum(prior_1[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.d_model)),
                        series_1[u].detach())

            metric = torch.softmax(
                (-series_loss - prior_loss) * self.r + (1 - self.r) * (-series_loss_1 - prior_loss_1), dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)

        pred = (test_energy > thresh).astype(int)
        gt = test_labels.astype(int)

        matrix = [self.index]
        scores_simple = combine_all_evaluation_scores(pred, gt, test_energy)
        for key, value in scores_simple.items():
            matrix.append(value)
            print('{0:21} : {1:0.4f}'.format(key, value))

        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)
        np.savetxt('pred.txt', pred, fmt='%d', delimiter='\n')
        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(accuracy, precision,
                                                                                                   recall, f_score))

        if self.data_path == 'UCR' or 'UCR_AUG':
            import csv
            with open('result/' + self.data_path + '.csv', 'a+') as f:
                writer = csv.writer(f)
                writer.writerow(matrix)

        return accuracy, precision, recall, f_score
