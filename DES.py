import numpy as np


class DES:
    def __init__(self, key, IV):
        # print(len(key))
        assert len(key) == 64
        self.key = self.curr_key = key
        self.IV = IV

        self.perm_table = [
            [58, 50, 42, 34, 26, 18, 10, 2],
            [60, 52, 44, 36, 28, 20, 12, 4],
            [62, 54, 46, 38, 30, 22, 14, 6],
            [64, 56, 48, 40, 32, 24, 16, 8],
            [57, 49, 41, 33, 25, 17, 9, 1],
            [59, 51, 43, 35, 27, 19, 11, 3],
            [61, 53, 45, 37, 29, 21, 13, 5],
            [63, 55, 47, 39, 31, 23, 15, 7],
        ]

        self.f_perm_table = [
            [40, 8, 48, 16, 56, 24, 64, 32],
            [39, 7, 47, 15, 55, 23, 63, 31],
            [38, 6, 46, 14, 54, 22, 62, 30],
            [37, 5, 45, 13, 53, 21, 61, 29],
            [36, 4, 44, 12, 52, 20, 60, 28],
            [35, 3, 43, 11, 51, 19, 59, 27],
            [34, 2, 42, 10, 50, 18, 58, 26],
            [33, 1, 41, 9, 49, 17, 57, 25],
        ]

        self.expansion_table = [
            [32, 1, 2, 3, 4, 5],
            [4, 5, 6, 7, 8, 9],
            [8, 9, 10, 11, 12, 13],
            [12, 13, 14, 15, 16, 17],
            [16, 17, 18, 19, 20, 21],
            [20, 21, 22, 23, 24, 25],
            [24, 25, 26, 27, 28, 29],
            [28, 29, 30, 31, 32, 1],
        ]

        self.s_boxs = {
            'S1':
            [
              [14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
              [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
              [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
              [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13],
              ],
            'S2':
            [
              [15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
              [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
              [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
              [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9],
              ],
            'S3':
            [
              [10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
              [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
              [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
              [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12],
              ],
            'S4':
            [
              [7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
              [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
              [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
              [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14],
              ],
            'S5':
            [
              [2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
              [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
              [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
              [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3],
              ],
            'S6':
            [
              [12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
              [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
              [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
              [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13],
              ],
            'S7':
            [
              [4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
              [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
              [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
              [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12],
              ],
            'S8':
            [
              [13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
              [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
              [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
              [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11],
              ]
        }

        self.round_keys = []

    def _binArrAdd(self, arr, n):
        Int = 0
        for i in range(len(arr)):
            Int += arr[i] * 2**i
        Int += n

        result = list(bin(Int))
        result = result[-len(arr):]
        result = map(lambda x: int(x), result)
        return list(result)

    def permutation(self, data, final=False):
        result = data.copy()
        table = self.f_perm_table
        if not final:
            table = self.perm_table

        for i in range(64):
            pos = table[int(i / 8)][i % 8] - 1
            result[pos] = data[i]

        return result

    def getRoundKey(self, reverse=False):
        # left_permu + right_permu = P1
        left_permu = [
            [57, 49, 41, 33, 25, 17, 9],
            [1, 58, 50, 42, 34, 26, 18],
            [10, 2, 59, 51, 43, 35, 27],
            [19, 11, 3, 60, 52, 44, 36],
        ]

        right_permu = [
            [63, 55, 47, 39, 31, 23, 15],
            [7, 62, 54, 46, 38, 30, 22],
            [14, 6, 61, 53, 45, 37, 29],
            [21, 13, 5, 28, 20, 12, 4],
        ]

        P_2 = [
            [14, 17, 11, 24, 1, 5],
            [3, 28, 15, 6, 21, 10],
            [23, 19, 12, 4, 26, 8],
            [16, 7, 27, 20, 13, 2],
            [41, 52, 31, 37, 47, 55],
            [30, 40, 51, 45, 33, 48],
            [44, 49, 39, 56, 34, 53],
            [46, 42, 50, 36, 29, 32],
        ]

        L = R = [-1 for i in range(28)]
        for i in range(28):
            L[i] = self.curr_key[left_permu[int(i / 7)][i % 7] - 1]
            R[i] = self.curr_key[right_permu[int(i / 7)][i % 7] - 1]

        bitshift = [0, 1, 8, 15] if not reverse else [1, 8, 15]
        for r in range(16):
            if not reverse:
                if r in bitshift:
                    L = L[1:] + [L[0]]
                    R = R[1:] + [R[0]]
                else:
                    L = L[2:] + L[:2]
                    R = R[2:] + R[:2]
            elif reverse and r != 0:
                if 16 - r in bitshift:
                    L = [L[-1]] + L[1:]
                    R = [R[-1]] + R[1:]
                else:
                    L = L[:-2] + L[2:]
                    R = R[:-2:] + R[2:]
            output = [-1] * 48
            temp = L + R

            for i in range(len(P_2) * len(P_2[0])):
                output[i] = temp[P_2[int(i / 6)][i % 6] - 1]

            yield output

    def F(self, R, round_key):
        r_permu_table = [
            [16, 7, 20, 21, 29, 12, 28, 17],
            [1, 15, 23, 26, 5, 18, 31, 10],
            [2, 8, 24, 14, 32, 27, 3, 9],
            [19, 13, 30, 6, 22, 11, 4, 25],
        ]

        R_expan = [-1] * 48
        for i in range(len(self.expansion_table) * len(self.expansion_table[0])):
            R_expan[i] = R[self.expansion_table[int(i / 6)][i % 6] - 1]

        xored_R = list(map(lambda x: x[0] ^ x[1], zip(R_expan, round_key)))

        bin32 = []
        for i in range(8):
            div = xored_R[i * 6:i * 6 + 6]
            box = self.s_boxs['S%d' % (i + 1)]

            row_id = div[0] * 2 + div[-1]
            col_id = ''
            for dig in div[1:-1]:
                col_id += str(dig)
            col_id = int(col_id, 2)

            N = box[row_id][col_id]
            bin4 = [int(x) for x in list('{0:0b}'.format(N))]
            bin4 = ([0] * (4 - len(bin4))) + bin4
            bin32 += bin4

        shuf_R = [-1] * 32
        for i in range(len(r_permu_table) * len(r_permu_table[0])):
            shuf_R[i] = bin32[r_permu_table[int(i / 8)][i % 8] - 1]

        return shuf_R

    def singleRound(self, L, R, round_key):
        assert len(L) == 32
        assert len(R) == 32

        shuf_R = self.F(R, round_key)
        new_R = list(map(lambda x: x[0] ^ x[1], zip(shuf_R, L)))
        new_L = R
        return new_L + new_R


    def _encrpy(self, data):
        enc_block = self.permutation(data).tolist()
        for key in self.getRoundKey():
            enc_block = self.singleRound(enc_block[:32], enc_block[32:], key)
        enc_block = self.permutation(enc_block[32:] + enc_block[:32], final=True)

        return np.array(enc_block)

    def _decrpy(self, data):
        enc_block = self.permutation(data).tolist()
        keys = [key for key in self.getRoundKey()]
        keys.reverse()
        for key in keys:
            enc_block = self.singleRound(enc_block[:32], enc_block[32:], key)
        enc_block = self.permutation(enc_block[32:] + enc_block[:32], final=True)

        return np.array(enc_block)

    def encrpy(self, data, mode='ECB'):
        assert type(data) is np.ndarray
        assert data.shape[1] == 64

        output_blocks = []
        last_param = None
        XOR = lambda X, Y: np.array(list(map(lambda x: x[0] ^ x[1], zip(X, Y))))

        print("Encrpy %d Block in %s mode" % (data.shape[0], mode))
        for ind, block in zip(range(data.shape[0]), data):

            if mode == 'ECB':
                enc_block = self._encrpy(block)
            elif mode == 'CBC':
                if ind == 0:
                    fb = XOR(block, self.IV)
                else:
                    fb = XOR(block, output_blocks[-1])
                enc_block = self._encrpy(fb)
            elif mode == 'OFB':
                if last_param is None:
                    last_param = enc_block = self._encrpy(np.array(self.IV))
                else:
                    last_param = enc_block = self._encrpy(last_param)
                enc_block = XOR(block, enc_block)
            elif mode == 'CTR':
                new_V = self._binArrAdd(self.IV, ind)
                enc_block = XOR(block, self._encrpy(np.array(new_V)))
            else:
                raise ValueError("Operation Mode [%s] dont exist!" % mode)

            output_blocks.append(enc_block)

        return np.array(output_blocks)

    def decrpy(self, data, mode='ECB'):
        assert type(data) is np.ndarray
        assert data.shape[1] == 64

        output_blocks = []
        last_param = None
        XOR = lambda X, Y: np.array(list(map(lambda x: x[0] ^ x[1], zip(X, Y))))

        print("Decrpy %d Block in %s mode" % (data.shape[0], mode))
        for ind, block in zip(range(data.shape[0]), data):

            if mode == 'ECB':
                enc_block = self._decrpy(block)
            elif mode == 'CBC':
                enc_block = self._decrpy(block)
                if ind == 0:
                    enc_block = XOR(enc_block, self.IV)
                else:
                    enc_block = XOR(enc_block, data[ind - 1])
            elif mode == 'OFB':
                if last_param is None:
                    last_param = enc_block = self._encrpy(np.array(self.IV))
                else:
                    last_param = enc_block = self._encrpy(last_param)
                enc_block = XOR(block, enc_block)
            elif mode == 'CTR':
                new_V = self._binArrAdd(self.IV, ind)
                enc_block = XOR(block, self._encrpy(np.array(new_V)))

            output_blocks.append(enc_block)

        return np.array(output_blocks)
