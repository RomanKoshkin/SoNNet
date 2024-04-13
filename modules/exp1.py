from scipy.ndimage import gaussian_filter
import numpy as np
from tqdm import trange
from modules.environment import InfinitePong
from modules.constants import bar_format
from modules.utils import *
import copy


class Exp1(object):

    def __init__(self, m, NE, stim_electrodes, exp_id, row=5, col=19):
        self.action = 0
        self.NE = NE
        self.m = m
        self.stim_electrodes = stim_electrodes
        self.env = InfinitePong(visible=False)

        self.WgtEvo = []
        self.FEvo = []
        self.DEvo = []
        self.MOD = []

        self.brief_stim = self._makeBriefStim(row=row, col=col)
        self.start_t = copy.deepcopy(self.m.getState().t)
        self.exp_path = f'../experiments/{exp_id}/assets'
        os.makedirs(self.exp_path, exist_ok=True)

    def on_sensory(self, im, stepMsInt):
        x1 = np.zeros((self.NE,), dtype=np.float64)
        x1[:400] = im.flatten()
        self.m.setStimIntensity(x1)
        self.m.sim(stepMsInt * 100)
        return im

    def _snapshot(self, i, im, activity, action, reward, pbar):
        W = np.copy(self.m.getWeights())[:self.NE, :self.NE]
        w_, labels, counts, mod, newids = clusterize(W)  # <<<<<<< !!!!!!!!!!!
        t = self.m.getState().t
        pbar.set_description(
            f'Time : {t/1000:.2f} s, mod: {mod:.2f}, N: {len(np.unique(labels))}, a: {action}, rw: {reward}')

        self.WgtEvo.append(calcAssWgts(t, self.m, labels))
        self.FEvo.append(calcF(t, self.m, labels))
        self.DEvo.append(calcD(t, self.m, labels))
        self.MOD.append(mod)
        _, ax = plt.subplots(1, 2, figsize=(15, 5))
        _ = ax[0].imshow(im)
        _ = ax[1].imshow(activity)
        _title = f'{self.m.getState().t:.2f} step: {self.env.env.stepid} mod: {mod:.2f}'
        _ = ax[1].set_title(_title)
        _ = plt.savefig(f'{self.exp_path}/activity_{i:05d}.png')
        _ = plt.close()

    def _makeBriefStim(self, row=None, col=None):
        im = np.zeros((20, 20))
        im[row, col] = 4
        return gaussian_filter(im, sigma=1)

    def run(self, nsteps, stepMsInt=10):

        removeFilesInFolder(self.exp_path)  # delete old frames

        time_now = self.m.getState().t
        pbar = trange(nsteps, desc=f'Time : {time_now:.2f}', bar_format=bar_format)  # 2000 for 400 s
        for i in pbar:
            try:
                action = np.random.choice([-1, 0, 1])
                reward = 0

                # the stim will last for 50 ms and then go blank
                im = self.brief_stim
                if time_now - self.start_t < 50:
                    _ = self.on_sensory(im, stepMsInt)
                else:
                    im = im * 0
                    _ = self.on_sensory(im, stepMsInt)

                time_now = self.m.getState().t
                activity = self.m.getRecents()[:400].reshape(20, 20)

                self._snapshot(i, im, activity, action, reward, pbar)

            except KeyboardInterrupt:
                cprint('User interrupt', color='green')
                break
