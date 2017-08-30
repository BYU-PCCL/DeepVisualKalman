from __future__ import print_function
from torch.autograd import Variable
from tqdm import tqdm
import utilities
from collections import defaultdict

def train(epoch, model, data_loader, optimizer, logger, args):
    model.train()
    with tqdm(enumerate(data_loader), total=len(data_loader), leave=False) as progress:
        for batch_idx, (data, target) in progress:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()

            output = model(data)

            loss, stats = model.loss(output, target)
            loss.backward()
            optimizer.step()

            logger.from_stats(stats, index=epoch * len(data_loader) + batch_idx)
            progress.set_description(utilities.stats_to_string(stats))

def test(epoch, model, data_loader, logger, args):
    model.eval()
    summarized_stats = defaultdict(float)
    with tqdm(enumerate(data_loader), total=len(data_loader), leave=False) as progress:
        for batch_idx, (data, target) in progress:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)

            output = model(data)
            loss, stats = model.loss(output, target)

            summarized_stats = utilities.add_stats(summarized_stats, stats)
            progress.set_description(utilities.stats_to_string(summarized_stats))

    summarized_stats = utilities.divide_stats_by_constant(summarized_stats, len(data_loader))

    return summarized_stats

if __name__ == '__main__':
    args = utilities.parse_arguments()

    with tqdm(range(1, args.epochs + 1)) as progress:
        for epoch in progress:
            train(epoch, args.model, args.train_loader, args.optimizer, args.train_logger, args)
            test_stats = test(epoch, args.model, args.test_loader, args.test_logger, args)

            # line up test statistics with the end of the train steps
            args.test_logger.from_stats(test_stats, index=(epoch + 1) * len(args.train_loader))

            progress.set_description(utilities.stats_to_string(test_stats, prepend='test_'))