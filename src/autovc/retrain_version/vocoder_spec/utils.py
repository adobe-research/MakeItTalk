import os

def _get_padding_conv2d(input_size, output_size, kernel_size, stride, dilation=[1,1]):
    Pr = (output_size[0]-1)*stride[0]+(kernel_size[0]-1)*dilation[0]+1-input_size[0]
    Pc = (output_size[1]-1)*stride[1]+(kernel_size[1]-1)*dilation[1]+1-input_size[1]
    padding_h = (Pr/2, Pr-Pr/2)
    padding_w = (Pc/2, Pc-Pc/2)
    print(padding_h, padding_w)


def _get_padding_deconv2d(input_size, output_size, kernel_size, stride):
    padding_h = (input_size[0]-1)*stride[0]+kernel_size[0]-output_size[0]
    padding_w = (input_size[1]-1)*stride[1]+kernel_size[1]-output_size[1]
    print(padding_h/2, padding_w/2)


def _conv2d_simulator(input_dim, kernel_size, stride, padding, dilation=[1,1]):
    h_out = (input_dim[0]+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0] + 1
    w_out = (input_dim[1]+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1] + 1
    print('Floor of:', h_out, w_out)
    
    
def _deconv2d_simulator(input_dim, kernel_size, stride, padding, dilation=[1,1]):
    h_out = (input_dim[0]-1)*stride[0]-2*padding[0]+kernel_size[0]
    w_out = (input_dim[1]-1)*stride[1]-2*padding[1]+kernel_size[1]
    print(h_out, w_out)
    


import numpy as np
import librosa
import pysptk
from scipy import signal
import pyworld as pw
import copy
import pdb

def sptk_left_signal_padding(x, count):
    x = np.pad(x, (count,0), 'constant', constant_values=(0, 0))
    return x

def sptk_frame_zero_padding(x, winsz):
    x = np.pad(x, ((0,0),(winsz//2,winsz//2)), 'constant', constant_values=(0, 0))
    return x

def sptk_signal_padding(x, count):
    x = np.pad(x, (count,count), 'constant', constant_values=(0, 0))
    return x

def sptk_window(x, framesz, hopsz, winsz=None, windowing=None, normalize=False):
    x = librosa.util.frame(sptk_signal_padding(x, framesz//2), frame_length=framesz, hop_length=hopsz)
    if windowing is not None:
        win = pysptk.blackman(framesz)
        x = x.T * win
    else:
        x = x.T
    if winsz is not None and winsz != framesz:
        x = sptk_frame_zero_padding(x, winsz-framesz)
    if normalize:
        x = x / np.sqrt(np.expand_dims(sum(x**2, 1), 1) + 1e-16)
    return x

def hz2alpha(hz):
    alpha = 0.313 * np.log10(hz) + (-0.903)
    alpha = np.round(alpha*100) / 100.0
    return alpha

def sptk_mcep(x, order, winsz, hopsz, fftsz, fs, window_norm=False, noise_floor=1e-8):
    alpha = hz2alpha(fs)
    windowed = sptk_window(x, winsz, hopsz, fftsz, windowing='blackman', normalize=window_norm)
    cep = pysptk.mcep(windowed, order=order, alpha=alpha, miniter=2, maxiter=30,
                      threshold=0.001, etype=1, eps=noise_floor, min_det=1.0e-6, itype=0)
    return cep, alpha    



def my_world(x, fs, fft_size=1024, hopsz=256, lo=50, hi=550):
    frame_period = hopsz / float(fs) * 1000
    _f0, t = pw.harvest(x, fs, frame_period=frame_period, f0_floor=lo, f0_ceil=hi)
    f0 = pw.stonemask(x, _f0, t, fs)
    sp = pw.cheaptrick(x, f0, t, fs, fft_size=fft_size, f0_floor=lo)
    ap = pw.d4c(x, f0, t, fs, fft_size=fft_size)
    assert x.shape[0] >= (sp.shape[0]-1) * hopsz
    sig = x[:(sp.shape[0]-1) * hopsz]
    assert sig.shape[0] % hopsz == 0
    return f0[:-1], sp[:-1,:], ap[:-1,:], sig



def global_normalization(x, lo, hi):
    # normalize logf0 to [0,1]
    x = x.astype(float).copy()
    uv = x==0
    x[~uv] = (x[~uv] - np.log(lo)) / (np.log(hi)-np.log(lo))
    x = np.clip(x, 0, 1)
    return x


def speaker_normalization(f0, index_nonzero, mean_f0, std_f0):
    # f0 is logf0
    f0 = f0.astype(float).copy()
    #index_nonzero = f0 != 0
    f0[index_nonzero] = (f0[index_nonzero] - mean_f0) / std_f0 / 4.0
    f0[index_nonzero] = np.clip(f0[index_nonzero], -1, 1)
    f0[index_nonzero] = (f0[index_nonzero] + 1) / 2.0
    return f0


def speaker_normalization_tweak(f0, mean_f0, std_f0, mean_f0_trg, std_f0_trg):
    # f0 is logf0
    f0 = f0.astype(float).copy()
    index_nonzero = f0 != 0
    delta = (mean_f0_trg - mean_f0) * 0.1
    f0[index_nonzero] = (f0[index_nonzero] - mean_f0 + delta) / std_f0 / 4.0 
    f0 = np.clip(f0, -1, 1)
    f0[index_nonzero] = (f0[index_nonzero] + 1) / 2.0
    return f0


def quantize_f0(x, num_bins=256):
    # x is logf0
    assert x.ndim==1
    x = x.astype(float).copy()
    assert (x >= 0).all() and (x <= 1).all()
    uv = x==0
    x = np.round(x * (num_bins-1))
    x = x + 1
    x[uv] = 0
    enc = np.zeros((len(x), num_bins+1), dtype=np.float32)
    enc[np.arange(len(x)), x.astype(np.int32)] = 1.0
    return enc 


def quantize_f0_interp(x, num_bins=256):
    # x is logf0
    assert x.ndim==1
    x = x.astype(float).copy()
    uv = (x<0)
    x[uv] = 0.0
    assert (x >= 0).all() and (x <= 1).all()
    x = np.round(x * (num_bins-1))
    x = x + 1
    x[uv] = 0.0
    enc = np.zeros((len(x), num_bins+1), dtype=np.float32)
    enc[np.arange(len(x)), x.astype(np.int32)] = 1.0
    return enc 


def quantize_chroma(x, lo=50, hi=400, num_bins=120):
    # x is f0 in Hz
    assert x.ndim==1
    x = x.astype(float).copy()
    uv = x==0
    x[~uv] = np.clip(x[~uv], lo/2, hi*2) 
    # convert to chroma f0
    x[~uv] = (np.log2(x[~uv] / 440) * 12 + 57) % 12
    # xs ~ [0,12)
    x = np.floor(x / 12 * num_bins)
    x = x + 1
    x[uv] = 0 
    enc = np.zeros((len(x), num_bins+1), dtype=np.float32)
    enc[np.arange(len(x)), x.astype(np.int32)] += 1.0
    
    return enc    



def quantize_f0s(xs, lo=50, hi=400, num_bins=256):
    # xs is logf0
    xs = copy.copy(xs)
    uv = xs==0
    xs[~uv] = (xs[~uv] - np.log(lo)) / (np.log(hi)-np.log(lo))
    xs = np.clip(xs, 0, 1)
    # xs ~ [0,1]
    xs = np.round(xs * (num_bins-1))
    xs = xs + 1
    xs[uv] = 0
    enc = np.zeros((xs.shape[1], num_bins+1), dtype=np.float32)
    for i in range(xs.shape[0]):
        enc[np.arange(xs.shape[1]), xs[i].astype(np.int32)] += 1.0
    enc /= enc.sum(axis=1, keepdims=True)    
    return enc  




def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def write_metadata(metadata, out_dir, sr=16000):
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write('|'.join([str(x) for x in m]) + '\n')
    frames = sum([m[2] for m in metadata])
    hours = frames / sr / 3600
    print('Wrote %d utterances, %d time steps (%.2f hours)' % (len(metadata), frames, hours))
    
    
def world_dio(x, fs, fft_size=1024, hopsz=256, lo=50, hi=550, thr=0.1):
    frame_period = hopsz / float(fs) * 1000
    _f0, t = pw.dio(x, fs, frame_period=frame_period, f0_floor=lo, f0_ceil=hi, allowed_range=thr)
    f0 = pw.stonemask(x, _f0, t, fs)
    f0[f0!=0] = np.log(f0[f0!=0])
    return f0


def world_harvest(x, fs, fft_size=1024, hopsz=256, lo=50, hi=550):
    frame_period = hopsz / float(fs) * 1000
    _f0, t = pw.harvest(x, fs, frame_period=frame_period, f0_floor=lo, f0_ceil=hi)
    f0 = pw.stonemask(x, _f0, t, fs)
    f0[f0!=0] = np.log(f0[f0!=0])
    return f0

import torch
def get_mask_from_lengths(lengths, max_len):
    ids = torch.arange(0, max_len, device=lengths.device)
    mask = (ids >= lengths.unsqueeze(1)).byte()
    return mask
    
    
def pad_sequence_cnn(sequences, padding_value=0):

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    channel_dim = max_size[0]
    max_len = max([s.size(-1) for s in sequences])
    
    out_dims = (len(sequences), channel_dim, max_len)
    
    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(-1)
        # use index notation to prevent duplicate references to the tensor
        out_tensor[i, :, :length] = tensor
    
    return out_tensor    



def interp_vector(vec, t_new):
    t = np.arange(vec.shape[0])
    out = np.zeros_like(vec)
    for j in range(vec.shape[1]):
        out[:,j] = np.interp(t_new, t, vec[:,j], left=np.nan, right=np.nan)
    assert not np.isnan(out).any()    
    return out



from scipy.interpolate import interp1d

def interp_vector_scipy(vec, t_new):
    t = np.arange(vec.shape[0])
    f_interp = interp1d(t, vec, axis=0, bounds_error=True, assume_sorted=True)
    out = f_interp(t_new)
    return out.astype(np.float32)
    
    
    