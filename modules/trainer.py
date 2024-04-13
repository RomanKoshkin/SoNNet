from scipy.ndimage import gaussian_filter
import numpy as np
from tqdm import trange
from modules.environment import InfinitePong
from modules.constants import bar_format
from modules.utils import *
from modules.video_utils import make_frame
import threading


class Trainer(object):

    def __init__(self, m, NE, NM, stim_electrodes, config=None):
        self.NE = NE
        self.NM = NM
        self.m = m
        self.stim_electrodes = stim_electrodes
        self.env = InfinitePong(visible=False, config=config)
        self.decision_margin = config['decision_margin'] if config is not None else 0

        self.xpos, self.ypos, self.reward, self.paddle_ymid = 0, 0, 0, 0
        self.XPOS, self.YPOS, self.PADDLE_YMID = [], [], []
        self.WgtEvo = []
        self.FEvo = []
        self.DEvo = []
        self.MOD = []
        self.prev_degree = None
        removeFilesInFolder('assets/')  # delete old frames
        removeFilesInFolder('tmp/')  # delete old frames
        self.config = config
        self.UP, self.DOWN, self.T, self.ACTION, = [], [], [], []
        self.REWARD, self.REWARD_T = [], []

    def update_state_memory(self):
        self.REWARD_T.append(self.m.getState().t)
        self.REWARD.append(self.reward)
        up, down, _ = self.get_action()
        self.UP.append(up)
        self.DOWN.append(down)
        self.T.append(self.m.getState().t)
        self.ACTION.append(self.action)
        self.XPOS.append(self.xpos)
        self.YPOS.append(self.ypos)
        self.PADDLE_YMID.append(self.paddle_ymid)

    def on_reward(self):
        x = np.ones((self.NE,)).astype('int32')
        self.m.setStim(x)

        if self.reward == -1:
            # FIXME: this must be an UNpredictable stim
            stimpat = np.zeros((400,), dtype=np.float64)
            if len(self.stim_electrodes) > 3:
                stim_electrodes = np.random.choice(self.stim_electrodes, 4, replace=False)
                stimpat[stim_electrodes] = 2
                stimpat = gaussian_filter(stimpat.reshape(20, 20), sigma=1)
            else:
                pass

            for i in range(10):
                x1 = np.zeros((self.NE,))
                x1[:400] = stimpat.flatten()
                self.m.setStimIntensity(x1)
                self.m.sim(5000)
                self.update_state_memory()
                self._snapshot(x1[:400].reshape(20, 20))

                x1 = np.zeros((self.NE,))
                self.m.setStimIntensity(x1)
                self.m.sim(5000)
                self.update_state_memory()
                self._snapshot(x1[:400].reshape(20, 20))
            return stimpat

        if self.reward == 1:
            # FIXME: this must be a predictable stim
            stimpat = np.zeros((400,), dtype=np.float64)
            if len(self.stim_electrodes) > 3:
                stim_electrodes = np.arange(3, 400, 6)
                stimpat[stim_electrodes] = 2
                stimpat = gaussian_filter(stimpat.reshape(20, 20), sigma=1.0)
            else:
                pass

            for i in range(1):
                x1 = np.zeros((self.NE,))
                x1[:400] = stimpat.flatten()
                self.m.setStimIntensity(x1)
                self.m.sim(5000)
                self.update_state_memory()
                self._snapshot(x1[:400].reshape(20, 20))

                x1 = np.zeros((self.NE,))
                self.m.setStimIntensity(x1)
                self.m.sim(5000)
                self.update_state_memory()
                self._snapshot(x1[:400].reshape(20, 20))
            return stimpat
        return None

    def on_sensory(self, im):
        x1 = np.zeros((self.NE,), dtype=np.float64)
        x1[:400] = im.flatten()
        self.m.setStimIntensity(x1)
        # simulate for 5000 ms, but take a snapshot every 500 ms
        for i in range(1):
            self.m.sim(5000)
            self.update_state_memory()
            if self.reward == 1:
                pass
            self._snapshot(im)
        return im

    def _snapshot(self, im):
        activity = self.m.getRecents()[:400].reshape(20, 20)
        W = np.copy(self.m.getWeights())[:self.NE, :self.NE]
        # w_, labels, counts, mod, newids = clusterize(W)  # <<<<<<< !!!!!!!!!!!
        # t = self.m.getState().t
        # self.pbar.set_description(
        #     f'Time : {t/1000:.2f} s, mod: {mod:.2f}, N: {len(np.unique(labels))}, a: {self.action}, rw: {self.reward}')

        # self.WgtEvo.append(calcAssWgts(t, self.m, labels))
        # self.FEvo.append(calcF(t, self.m, labels))
        # self.DEvo.append(calcD(t, self.m, labels))
        # self.MOD.append(mod)

        t = self.m.getState().t
        self.pbar.set_description(f'Time : {t/1000:.2f} s, a: {self.action}, rw: {self.reward}')

        indegrees = W[:400, :400].sum(axis=1).reshape(20, 20)
        outdegrees = W[:400, :400].sum(axis=0).reshape(20, 20)
        degree = indegrees + outdegrees
        if self.prev_degree is None:
            degree_diff = np.ones((20, 20)) * 1e-9
        else:
            degree_diff = degree - self.prev_degree
        self.prev_degree = np.copy(degree)

        X, Y, U, V, XX, YY, II, JJ, Z = getVF(W[:400, :400])

        normalized_degree = degree / degree.max()
        derivativeOrDegree = degree_diff / degree_diff.max()

        state_dict = dict(
            im=im,
            activity=activity,
            UP=self.UP,
            DOWN=self.DOWN,
            T=self.T,
            ACTION=self.ACTION,
            REWARD_T=self.REWARD_T,
            REWARD=self.REWARD,
            XPOS=self.XPOS,
            YPOS=self.YPOS,
            PADDLE_YMID=self.PADDLE_YMID,
            reward=self.reward,
            action=self.action,
            normalized_degree=normalized_degree,
            derivativeOrDegree=derivativeOrDegree,
            X=X,
            Y=Y,
            U=U,
            V=V,
            XX=XX,
            YY=YY,
            II=II,
            JJ=JJ,
            Z=Z,
            stepid=self.env.env.stepid,
            t=self.m.getState().t,
        )
        fname = f"tmp/state_dict_{int(self.m.getState().t*100):09d}.p"
        with open(fname, "wb") as f:
            pickle.dump(state_dict, file=f)
        # threading.Thread(group=None, target=make_frame, args=(fname,)).start()

    def get_action(self):
        # NOTE: get the recent activity of motor nerons
        motor_activity = self.m.getRecents()[self.NE - self.NM:self.NE].copy()
        up, down = (i.mean() for i in np.split(motor_activity, 2))
        up, down = self.softmax([up, down])
        go_up = up - down > self.decision_margin
        go_down = (down - up > self.decision_margin)
        if go_up:
            return up, down, -1
        elif go_down:
            return up, down, 1
        else:
            return up, down, 0

    @staticmethod
    def softmax(x):
        return np.exp(x) / sum(np.exp(x))

    def train(self, nsteps):

        self.pbar = trange(nsteps, desc=f'Time : {self.m.getState().t:.2f}', bar_format=bar_format)  # 2000 for 400 s
        for self.i in self.pbar:
            try:

                # self.action = np.random.choice([-1, 0, 1])
                _, _, self.action = self.get_action()
                im, self.xpos, self.ypos, self.reward, self.paddle_ymid = self.env.step(
                    action=self.action,
                    gauss=True,
                )

                if self.reward == 0:
                    _ = self.on_sensory(im)
                else:
                    _ = self.on_reward()

            except KeyboardInterrupt:
                cprint('User interrupt', color='green')
                break
