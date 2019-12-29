import numpy as np
import abc

import matplotlib.pyplot as plt

EPS = 8.854187817e-12
MU = 4 * np.pi * 1e-7
C = 1 / np.sqrt(MU * EPS)
DS = 7.5 * 1e-3
DT = DS / 3 / C


class TE:
    @abc.abstractmethod
    def cmp_Ex(self):
        pass

    @abc.abstractmethod
    def cmp_Ey(self):
        pass

    @abc.abstractmethod
    def cmp_Hz(self):
        pass

    @abc.abstractmethod
    def bdy_PEC(self):
        pass

    @abc.abstractmethod
    def ext_src(self, n):
        pass

    @abc.abstractmethod
    def save(self, n):
        pass

    @abc.abstractmethod
    def run(self):
        pass


class FCC_TE(TE):
    def __init__(self, lx, ly, nt, src_a, src_freq, src_phrase, src_type):

        self.nx = int(lx / DS) * 2
        self.ny = int(ly / DS) * 2
        self.nt = nt

        self.src_x = int(self.nx / 2)
        self.src_y = int(self.ny / 2)
        self.src_a = src_a
        self.src_freq = src_freq
        self.src_phrase = src_phrase
        self.src_type = src_type

        self.CA = 1
        self.CB = DT / (DS * EPS)
        self.CP = 1
        self.CQ = DT / (DS * MU)

        self.Ex = np.zeros((self.nx + 1, self.ny + 1))
        self.Ey = np.zeros((self.nx + 1, self.ny + 1))
        self.Hz = np.zeros((self.nx, self.ny))

        self.l_Hz = []

    def cmp_Ex(self):
        for i in range(1, self.nx):
            for j in range(1, self.ny):
                self.Ex[i][j] = self.CA * self.Ex[i][j] + \
                                self.CB * (self.Hz[i][j] + self.Hz[i - 1][j] - self.Hz[i][j - 1] - self.Hz[i - 1][
                    j - 1])

    def cmp_Ey(self):
        for i in range(1, self.nx):
            for j in range(1, self.ny):
                self.Ey[i][j] = self.CA * self.Ey[i][j] - \
                                self.CB * (self.Hz[i][j] + self.Hz[i][j - 1] - self.Hz[i - 1][j] - self.Hz[i - 1][
                    j - 1])

    def cmp_Hz(self):
        for i in range(self.nx):
            for j in range(self.ny):
                v1 = self.Ey[i + 1][j + 1] + self.Ey[i + 1][j] - self.Ey[i][j + 1] - self.Ey[i][j]
                v2 = self.Ex[i][j + 1] + self.Ex[i + 1][j + 1] - self.Ex[i][j] - self.Ex[i + 1][j]
                self.Hz[i][j] = self.CP * self.Hz[i][j] - self.CQ * (v1 - v2)

    def bdy_PEC(self):
        for i in range(self.nx + 1):
            self.Ex[i][0] = 0
            self.Ex[i][self.ny] = 0
        for i in range(self.ny + 1):
            self.Ey[0][i] = 0
            self.Ey[self.nx][i] = 0

    def ext_src(self, n):
        if self.src_type == 'Gaussian':
            tau = 30 * DT
            t0 = 0.8 * tau
            tmp = (n * DT - t0) / tau
            self.Hz[self.src_x][self.src_y] += np.cos(4 * np.pi * 1e10 * n * DT) * self.src_a * np.exp(
                -4 * np.pi * tmp * tmp)
        if self.src_type == 'Sin':
            self.Hz[self.src_x][self.src_y] = self.src_a * np.sin(
                2 * np.pi * self.src_freq * n * DT + self.src_phrase)

    def save(self, n):
        filename = 'data/FCC_{0:{1}}.txt'.format(n, '05d')
        with open(filename, 'w') as f:
            for i in range(0, self.nx - 1, 2):
                for j in range(0, self.ny - 1, 2):
                    f.write(str(
                        self.Hz[i][j] + self.Hz[i][j + 1] + self.Hz[i + 1][j] + self.Hz[i + 1][j + 1]) + '\t')
                f.write('\n')

    def run(self):
        print('Running FCC-FDTD')
        for i in range(1, self.nt + 1):
            self.cmp_Ex()
            self.cmp_Ey()
            self.ext_src(i)
            self.bdy_PEC()
            self.cmp_Hz()
            self.l_Hz.append(self.Hz[4][4] + self.Hz[5][4] + self.Hz[4][5] + self.Hz[5][5])
            if i % 10 == 0:
                self.save(i)
                print('steps:', str(i))


class YEE_TE(TE):
    def __init__(self, lx, ly, nt, src_a, src_freq, src_phrase, src_type):

        self.nx = int(lx / DS)
        self.ny = int(ly / DS)
        self.nt = nt

        self.src_x = int(self.nx / 2)
        self.src_y = int(self.ny / 2)
        self.src_a = src_a
        self.src_freq = src_freq
        self.src_phrase = src_phrase
        self.src_type = src_type

        self.CA = 1
        self.CB = DT / (DS * EPS)
        self.CP = 1
        self.CQ = DT / (DS * MU)

        self.Ex = np.zeros((self.nx, self.ny + 1))
        self.Ey = np.zeros((self.nx + 1, self.ny))
        self.Hz = np.zeros((self.nx, self.ny))

        self.l_Hz = []

    def cmp_Ex(self):
        for i in range(self.nx):
            for j in range(1, self.ny):
                self.Ex[i][j] = self.CA * self.Ex[i][j] + \
                                self.CB * (self.Hz[i][j] - self.Hz[i][j - 1])

    def cmp_Ey(self):
        for i in range(1, self.nx):
            for j in range(self.ny):
                self.Ey[i][j] = self.CA * self.Ey[i][j] - \
                                self.CB * (self.Hz[i][j] - self.Hz[i - 1][j])

    def cmp_Hz(self):
        for i in range(self.nx):
            for j in range(self.ny):
                v1 = self.Ey[i + 1][j] - self.Ey[i][j]
                v2 = self.Ex[i][j + 1] - self.Ex[i][j]
                self.Hz[i][j] = self.CP * self.Hz[i][j] - self.CQ * (v1 - v2)

    def bdy_PEC(self):
        for i in range(self.nx):
            self.Ex[i][0] = 0
            self.Ex[i][self.ny] = 0
        for i in range(self.ny):
            self.Ey[0][i] = 0
            self.Ey[self.nx][i] = 0

    def ext_src(self, n):
        if self.src_type == 'Gaussian':
            # self.Hz[self.src_x][self.src_y] += np.cos(4 * np.pi* 1e-6*n * DT) * self.src_a * np.exp(
            #     -4 * np.pi * tmp * tmp)
            # self.Hz[self.src_x][self.src_y] += self.src_a * np.exp(
            #     -4 * np.pi * tmp * tmp)
            tau = 40 * DT
            t0 = 3 * tau
            tmp = (n * DT - t0) / tau
            self.Hz[self.src_x][self.src_y] += self.src_a * np.exp(-4 * np.pi * tmp * tmp)
        if self.src_type == 'Sin':
            self.Hz[self.src_x][self.src_y] = self.src_a * np.sin(
                2 * np.pi * self.src_freq * n * DT + self.src_phrase)

    def save(self, n):
        filename = 'data/YEE_{0:{1}}.txt'.format(n, '05d')
        with open(filename, 'w') as f:
            for i in range(self.nx):
                for j in range(self.ny):
                    f.write(str(self.Hz[i][j]) + '\t')
                f.write('\n')

    def run(self):
        print('Running YEE-FDTD')
        for i in range(1, self.nt + 1):
            self.cmp_Ex()
            self.cmp_Ey()
            self.ext_src(i)
            self.bdy_PEC()
            self.cmp_Hz()
            self.l_Hz.append(self.Hz[2][2])
            if i % 10 == 0:
                self.save(i)
                print('steps:', str(i))


if __name__ == '__main__':
    yee = YEE_TE(.375, .375, 2000, 1, 1e9, 0, 'Sin')
    yee.run()
    np.savetxt('Hz_TE_YEE', np.array(yee.l_Hz))
    #
    fcc = FCC_TE(.375, .375, 2000, 1, 1e9, 0, 'Sin')
    fcc.run()
    np.savetxt('Hz_TE_FCC', np.array(fcc.l_Hz))
