import os
import sys
import pandas as pd
from matplotlib import pyplot as plt

import utils

def main(args):
    parser = utils.get_basic_parser()
    opt = parser.parse_args(args)

    batch_sizes = []

    for epoch in range(opt.n_epochs):
        batch_sizes.append(utils.get_batch_sizes(opt, epoch))

    df = pd.DataFrame(batch_sizes, columns=['real_batch', 'fake_batch'])
    max_size = max(df.max())
    ax = df.plot(ylim=(0, int(1.05 * max_size)),
            xlim=(0, opt.n_epochs),
            alpha=0.6)
    ax.set_xlabel("epoch")
    plt.savefig("{}_{}_{}_{}.png".format(
                opt.policy, opt.batch_size, opt.batch_interval, opt.n_epochs),
                dpi=1000)

if __name__ == "__main__":
    main(sys.argv[1:])