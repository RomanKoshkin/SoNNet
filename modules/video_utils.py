import sys

sys.path.append('../')
from multiprocessing import Pool
import pickle, os
import matplotlib.pyplot as plt
try:
    from constants import bar_format
except:
    from modules.constants import bar_format
from tqdm import tqdm


def make_frame(fname):
    with open(fname, 'rb') as f:
        d = pickle.load(f)
    gs_kw = dict(width_ratios=[1, 1, 1, 1], height_ratios=[5, 1])
    fig, ax = plt.subplots(2, 4, figsize=(17, 7), gridspec_kw=gs_kw)

    # plot motor activity
    _ = ax[1, 0].plot(d['T'], d['UP'], lw=1)
    _ = ax[1, 0].plot(d['T'], d['DOWN'], lw=1)
    _ = ax[1, 0].axhline(0.5, lw=1, ls=':')
    _ = ax[1, 0].set_ylim(0.48, 0.52)

    # plot action
    _ = ax[1, 1].plot(d['T'], d['ACTION'], lw=1, label='action')
    _ = ax[1, 1].axhline(0, lw=1, ls=':')
    _ = ax[1, 1].set_ylim(-1.2, 1.2)
    ax[1, 1].legend(loc='upper left')

    # plot reward
    axt = ax[1, 1].twinx()
    axt.plot(d['REWARD_T'], d['REWARD'], color='red', label='reward')
    axt.set_ylim(-1.2, 1.2)
    axt.legend(loc='upper right')

    _ = ax[0, 0].imshow(d['im'])
    _ = ax[0, 0].set_title(f'stimulus')

    ax[0, 1].quiver(
        d['X'],
        d['Y'],
        d['U'],
        d['V'],
        color='red',
        width=0.01,
    )
    ax[0, 1].quiver(
        d['XX'].flatten(),
        d['YY'].flatten(),
        d['II'].flatten(),
        d['JJ'].flatten(),
        color='yellow',
        alpha=0.3,
        width=0.005,
    )
    _ = ax[0, 1].imshow(d['activity'])

    ax[0, 1].grid()
    ax[0, 1].set_xticks([0., 5., 10., 15., 20.])
    ax[0, 1].set_yticks([0., 5., 10., 15., 20.])
    ax[0, 1].set_xlim(0, 19.5)
    ax[0, 1].set_ylim(0, 19.5)
    ax[0, 1].invert_yaxis()
    ax[0, 1].set_title('low-passed activity')

    _ = ax[0, 2].imshow(d['normalized_degree'])
    _ = ax[0, 2].set_title('degree')

    _ = ax[0, 3].imshow(d['derivativeOrDegree'])
    _ = ax[0, 3].set_title('deriv. of degree')

    _title = (f"{d['t']:.2f} step: {d['stepid']}' + f' action {d['action']}, rw: {d['reward']}")
    _ = ax[0, 1].set_title(_title)
    _ = plt.savefig(f"../assets/activity_{int(d['t']*100):09d}.png", dpi=200)
    _ = plt.close()


if __name__ == "__main__":
    fnames = [f"../tmp/{i}" for i in os.listdir('../tmp/') if i.startswith('state_dict')]
    with Pool(25) as p:
        res = list(tqdm(
            p.imap(make_frame, fnames),
            total=len(fnames),
            bar_format=bar_format,
        ))
