from PIL import Image
import numpy as np
from DES import *
import time

def toBinArray(B):
    raw_array = np.frombuffer(B, dtype=np.uint8)
    out = np.unpackbits(raw_array)
    if len(out) % 64 != 0:
        out = np.concatenate([out, np.zeros(64 - len(out) % 64, dtype=np.uint8)])
    return out.reshape([-1, 64])

def toBytes(Ba, ori_bit_len):
    temp = Ba.reshape([-1])
    temp = temp[:ori_bit_len]
    byte_arr = np.packbits(temp)
    byte_arr = byte_arr.tobytes()

    return  byte_arr

inputfile = 'test.bmp'
img = Image.open('test.bmp')

raw = img.tobytes()
raw_array = toBinArray(raw)

bands  = img.split()
raw_band_byte = bands[0].tobytes()
raw_R = toBinArray(bands[0].tobytes())
raw_G = toBinArray(bands[1].tobytes())
raw_B = toBinArray(bands[2].tobytes())


key = [1, 0, 1, 0, 1, 0, 1, 1] * 8
IV = [1, 1, 1, 1, 1, 0, 1, 1] * 8

des = DES(key, IV)
for mode in ['ECB', 'CBC', 'OFB', 'CTR']:
    start_time = time.time()
    cyper_text = des.encrpy(raw_array, mode=mode)
    b_cyper_text = toBytes(cyper_text, len(raw) * 8)
    cyper_img = Image.frombytes(img.mode, img.size, b_cyper_text)
    cyper_img.save('cry_%s.bmp' % mode)

    decyper_text = des.decrpy(cyper_text, mode=mode)
    b_decyper_text = toBytes(decyper_text, len(raw) * 8)
    decyper_img = Image.frombytes(img.mode, img.size, b_decyper_text)
    decyper_img.save('decry_%s.bmp' % mode)
    end_time = time.time()
    print("Time passed: %f sec" % (end_time - start_time))

for mode in ['ECB', 'CBC', 'OFB', 'CTR']:
    cyps = []
    out = []
    dec_out = []
    start_time = time.time()

    for data in [raw_R, raw_G, raw_B]:
        cyper_text = des.encrpy(data, mode=mode)
        cyps.append(cyper_text)
        b_cyper_text = toBytes(cyper_text, len(raw_band_byte) * 8)
        out.append(b_cyper_text)

    cyper_img = Image.merge("RGB", [Image.frombytes('L', bands[0].size, o) for o in out])
    cyper_img.save('cry_%s_RGB.bmp' % mode)

    for data, i in zip(cyps, range(len(cyps))):
        decyper_text = des.decrpy(data, mode=mode)
        b_decyper_text = toBytes(decyper_text, len(raw_band_byte) * 8)
        dec_out.append(b_decyper_text)
    decyper_img = Image.merge("RGB", [Image.frombytes('L', bands[0].size, o) for o in dec_out])
    decyper_img.save('decry_%s_RGB.bmp' % mode)
    end_time = time.time()
    print("Time passed: %f sec" % (end_time - start_time))
