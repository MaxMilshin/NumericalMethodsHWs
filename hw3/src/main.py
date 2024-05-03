#!/usr/bin/env python3

import os
import argparse
from PIL import Image
import numpy as np
from numpy.linalg import norm
from numpy import linalg
from random import normalvariate
from math import sqrt

import warnings

warnings.filterwarnings("ignore", category=np.ComplexWarning)

IMAGE_HEADER_SIZE = 138
NEW_IMAGE_HEADER_SIZE = 12
FLOAT_SIZEOF = 4
START_ERROR = 10000
EPS = 0.1


def get_new_rank(ncols, nrows, n):
    img_size = nrows * ncols * 3 + IMAGE_HEADER_SIZE
    new_img_size = img_size // n
    l = 0
    r = min(ncols, nrows) + 1
    while r - l > 1:
        m = (l + r) // 2
        if (ncols * m + m * m + nrows * m) * 3 * FLOAT_SIZEOF <= new_img_size:
            l = m 
        else:
            r = m 
    return l

def naive_svd(m, _):
    if m.shape[0] >= m.shape[1]:
        mtm = np.dot(m, m.T)
    else:
        mtm = np.dot(m.T, m)
    
    eigen_values, eigen_vectors = np.linalg.eig(mtm)
    
    sorted_indices = np.argsort(eigen_values)[::-1]
    eigen_values = eigen_values[sorted_indices]
    eigen_vectors = eigen_vectors[:, sorted_indices]
    
    sigma = np.sqrt(eigen_values)

    u = np.dot(m, eigen_vectors)
    u /= np.linalg.norm(u, axis=0) 
    
    v = eigen_vectors
    
    return u, sigma, v.T

def advanced_svd(m, k):
    height, width = m.shape

    def random_unit_vector(n):
        unnormalized = np.random.normal(size=n) 
        return unnormalized / np.linalg.norm(unnormalized)

    v = np.matrix([random_unit_vector(k) for _ in range(width)])
    delta = START_ERROR
    cnt = 0
    while delta > EPS:
        # if cnt % 1 == 0:
        #     print(delta)
        cnt += 1
        q, _ = np.linalg.qr(m @ v)
        u = np.matrix(q[:, 0:k])

        q, r = np.linalg.qr(m.T @ u)
        v = np.matrix(q[:, 0:k])
        sigma = np.matrix(r[0:k, 0:k])

        delta = np.linalg.norm(m @ v - u * sigma)
        if cnt == 10:
            break

    return (u, np.diagonal(sigma).astype(np.float32), v.T)

def compress_image(input_path, output_path, svd_method, N):
    img = Image.open(input_path)
    height, width = img.size
    k = get_new_rank(height, width, N)
    
    img_array = np.array(img, dtype='float64')
    
    red_channel = img_array[:,:,0]
    green_channel = img_array[:,:,1]
    blue_channel = img_array[:,:,2]

    U_r, Sigma_r, Vt_r = svd_method(red_channel, k)
    U_g, Sigma_g, Vt_g = svd_method(green_channel, k)
    U_b, Sigma_b, Vt_b = svd_method(blue_channel, k)
    
    U_r=U_r[:,:k]
    Sigma_r = Sigma_r[:k]
    Vt_r=Vt_r[:k,:]

    U_g=U_g[:,:k]
    Sigma_g = Sigma_g[:k]
    Vt_g=Vt_g[:k,:]
    
    U_b=U_b[:,:k] 
    Sigma_b = Sigma_b[:k]
    Vt_b=Vt_b[:k,:]

    
    data = bytearray()
    data.extend(np.uint32(width).tobytes())
    data.extend(np.uint32(height).tobytes())
    data.extend(np.uint32(k).tobytes())

    data.extend(U_r.astype(dtype=np.float32).tobytes())
    data.extend(Sigma_r.astype(dtype=np.float32).tobytes())
    data.extend(Vt_r.astype(dtype=np.float32).tobytes())

    data.extend(U_g.astype(dtype=np.float32).tobytes())
    data.extend(Sigma_g.astype(dtype=np.float32).tobytes())
    data.extend(Vt_g.astype(dtype=np.float32).tobytes())

    data.extend(U_b.astype(dtype=np.float32).tobytes())
    data.extend(Sigma_b.astype(dtype=np.float32).tobytes())
    data.extend(Vt_b.astype(dtype=np.float32).tobytes())

    with open(output_path, 'wb') as f:
        f.write(data)


def decompress_intermediate(input_path, output_path):
    with open(input_path, 'rb') as f:
        data = f.read()

    width = int(np.frombuffer(data, offset=0, dtype=np.uint32, count=1)[0])
    height = int(np.frombuffer(data, offset=4, dtype=np.uint32, count=1)[0])
    k = int(np.frombuffer(data, offset=8, dtype=np.uint32, count=1)[0])

    offset = NEW_IMAGE_HEADER_SIZE
    U_size = width * k
    Vt_size = k * height
    
    U_r = np.frombuffer(data, offset=offset, dtype=np.float32, count=U_size).reshape((width, k))
    offset += U_r.size * FLOAT_SIZEOF
    Sigma_r = np.diag(np.frombuffer(data, offset=offset, dtype=np.float32, count=k))
    offset += k * FLOAT_SIZEOF
    Vt_r = np.frombuffer(data, offset=offset, dtype=np.float32, count=Vt_size).reshape((k, height))
    offset += Vt_r.size * FLOAT_SIZEOF

    U_g = np.frombuffer(data, offset=offset, dtype=np.float32, count=U_size).reshape((width, k))
    offset += U_g.size * FLOAT_SIZEOF
    Sigma_g = np.diag(np.frombuffer(data, offset=offset, dtype=np.float32, count=k))
    offset += k * FLOAT_SIZEOF
    Vt_g = np.frombuffer(data, offset=offset, dtype=np.float32, count=Vt_size).reshape((k, height))
    offset += Vt_g.size * FLOAT_SIZEOF

    U_b = np.frombuffer(data, offset=offset, dtype=np.float32, count=U_size).reshape((width, k))
    offset += U_b.size * FLOAT_SIZEOF
    Sigma_b = np.diag(np.frombuffer(data, offset=offset, dtype=np.float32, count=k))
    offset += k * FLOAT_SIZEOF
    Vt_b = np.frombuffer(data, offset=offset, dtype=np.float32, count=Vt_size).reshape((k, height))
    offset += Vt_b.size * FLOAT_SIZEOF

    red_channel = U_r @ Sigma_r @ Vt_r
    green_channel = U_g @ Sigma_g @ Vt_g
    blue_channel = U_b @ Sigma_b @ Vt_b

    image_array = np.stack(
        (red_channel, green_channel, blue_channel), 
        axis=-1
    )

    decompressed_image = Image.fromarray(np.uint8(image_array))
    decompressed_image.save(output_path)
    

def parse_arguments():
    parser = argparse.ArgumentParser(description="Compressing and decompressing image using SVD")
    parser.add_argument("--mode", choices=["compress", "decompress"], required=True, help="Mode: compress or decompress")
    parser.add_argument("--method", choices=["numpy", "naive", "advanced"], help="SVD method (for compress mode only)")

    parser.add_argument("--compression_factor", type=int, help="Compression factor (for compress mode only)")
    parser.add_argument("--in_file", required=True, help="input file path")
    parser.add_argument("--out_file", required=True, help="output file path")

    args = parser.parse_args()

    if args.mode == "compress":
        if not args.method:
            parser.error("Method argument is required for compress mode")
        if not args.compression_factor:
            parser.error("Compression factor argument is required for compress mode")
    
    return args


if __name__ == "__main__":
    args = parse_arguments()
    if args.mode == "compress":
        svd_fs = {
            "numpy" : lambda m, _ : np.linalg.svd(m, full_matrices=False),
            "naive" : lambda m, _ : naive_svd(m, _),
            "advanced" : lambda m, k : advanced_svd(m, k)
        }
        compress_image(args.in_file, args.out_file, svd_fs[args.method], args.compression_factor)
        image_size = os.path.getsize(args.in_file)
    else:
        decompress_intermediate(args.in_file, args.out_file)