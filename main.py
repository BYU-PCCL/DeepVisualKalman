from __future__ import print_function
from torch.autograd import Variable
from tqdm import tqdm
import utilities


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
            logger.from_stats(stats)
            progress.set_description('Loss: {:.6f}'.format(loss.data[0]))

def test(epoch, model, data_loader, logger, args):
    model.eval()
    test_loss = 0
    correct = 0
    with tqdm(enumerate(data_loader), total=len(data_loader), leave=False) as progress:
        for batch_idx, (data, target) in progress:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)

            output = model(data)
            loss, stats = model.loss(output, target)
            test_loss += loss.data[0]  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(args.test_loader.dataset)
        progress.set_description('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(args.test_loader.dataset),
            100. * correct / len(args.test_loader.dataset)))

    logger.scalar('loss', loss, epoch)

    return test_loss

if __name__ == '__main__':
    args = utilities.parse_arguments()

    with tqdm(range(1, args.epochs + 1)) as progress:
        for epoch in progress:
            train(epoch, args.model, args.train_loader, args.optimizer, args.train_logger, args)
            test_loss = test(epoch, args.model, args.test_loader, args.test_logger, args)
            progress.set_description('Last Test Loss: {:.4f}'.format(test_loss))