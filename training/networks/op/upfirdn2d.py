import jittor as jt
import jittor.nn as nn


header = r"""
static __host__ __device__ __forceinline__ int floor_div(int a, int b) {
  int c = a / b;
  if (c * b > a) {
    c--;
  }
  return c;
}
struct UpFirDn2DKernelParams {
  int up_x;
  int up_y;
  int down_x;
  int down_y;
  int pad_x0;
  int pad_x1;
  int pad_y0;
  int pad_y1;
  int major_dim;
  int in_h;
  int in_w;
  int minor_dim;
  int kernel_h;
  int kernel_w;
  int out_h;
  int out_w;
  int loop_major;
  int loop_x;
};
using scalar_t = float;
template <typename scalar_t>
__global__ void upfirdn2d_kernel_large(scalar_t *out, const scalar_t *input,
                                       const scalar_t *kernel,
                                       const UpFirDn2DKernelParams p) {
  int minor_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int out_y = minor_idx / p.minor_dim;
  minor_idx -= out_y * p.minor_dim;
  int out_x_base = blockIdx.y * p.loop_x * blockDim.y + threadIdx.y;
  int major_idx_base = blockIdx.z * p.loop_major;
  if (out_x_base >= p.out_w || out_y >= p.out_h ||
      major_idx_base >= p.major_dim) {
    return;
  }
  int mid_y = out_y * p.down_y + p.up_y - 1 - p.pad_y0;
  int in_y = min(max(floor_div(mid_y, p.up_y), 0), p.in_h);
  int h = min(max(floor_div(mid_y + p.kernel_h, p.up_y), 0), p.in_h) - in_y;
  int kernel_y = mid_y + p.kernel_h - (in_y + 1) * p.up_y;
  for (int loop_major = 0, major_idx = major_idx_base;
       loop_major < p.loop_major && major_idx < p.major_dim;
       loop_major++, major_idx++) {
    for (int loop_x = 0, out_x = out_x_base;
         loop_x < p.loop_x && out_x < p.out_w; loop_x++, out_x += blockDim.y) {
      int mid_x = out_x * p.down_x + p.up_x - 1 - p.pad_x0;
      int in_x = min(max(floor_div(mid_x, p.up_x), 0), p.in_w);
      int w = min(max(floor_div(mid_x + p.kernel_w, p.up_x), 0), p.in_w) - in_x;
      int kernel_x = mid_x + p.kernel_w - (in_x + 1) * p.up_x;
      const scalar_t *x_p =
          &input[((major_idx * p.in_h + in_y) * p.in_w + in_x) * p.minor_dim +
                 minor_idx];
      const scalar_t *k_p = &kernel[kernel_y * p.kernel_w + kernel_x];
      int x_px = p.minor_dim;
      int k_px = -p.up_x;
      int x_py = p.in_w * p.minor_dim;
      int k_py = -p.up_y * p.kernel_w;
      scalar_t v = 0.0f;
      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          v += static_cast<scalar_t>(*x_p) * static_cast<scalar_t>(*k_p);
          x_p += x_px;
          k_p += k_px;
        }
        x_p += x_py - w * x_px;
        k_p += k_py - w * k_px;
      }
      out[((major_idx * p.out_h + out_y) * p.out_w + out_x) * p.minor_dim +
          minor_idx] = v;
    }
  }
}
template <typename scalar_t, int up_x, int up_y, int down_x, int down_y,
          int kernel_h, int kernel_w, int tile_out_h, int tile_out_w>
__global__ void upfirdn2d_kernel(scalar_t *out, const scalar_t *input,
                                 const scalar_t *kernel,
                                 const UpFirDn2DKernelParams p) {
  const int tile_in_h = ((tile_out_h - 1) * down_y + kernel_h - 1) / up_y + 1;
  const int tile_in_w = ((tile_out_w - 1) * down_x + kernel_w - 1) / up_x + 1;
  __shared__ volatile float sk[kernel_h][kernel_w];
  __shared__ volatile float sx[tile_in_h][tile_in_w];
  int minor_idx = blockIdx.x;
  int tile_out_y = minor_idx / p.minor_dim;
  minor_idx -= tile_out_y * p.minor_dim;
  tile_out_y *= tile_out_h;
  int tile_out_x_base = blockIdx.y * p.loop_x * tile_out_w;
  int major_idx_base = blockIdx.z * p.loop_major;
  if (tile_out_x_base >= p.out_w | tile_out_y >= p.out_h |
      major_idx_base >= p.major_dim) {
    return;
  }
  for (int tap_idx = threadIdx.x; tap_idx < kernel_h * kernel_w;
       tap_idx += blockDim.x) {
    int ky = tap_idx / kernel_w;
    int kx = tap_idx - ky * kernel_w;
    scalar_t v = 0.0;
    if (kx < p.kernel_w & ky < p.kernel_h) {
      v = kernel[(p.kernel_h - 1 - ky) * p.kernel_w + (p.kernel_w - 1 - kx)];
    }
    sk[ky][kx] = v;
  }
  for (int loop_major = 0, major_idx = major_idx_base;
       loop_major < p.loop_major & major_idx < p.major_dim;
       loop_major++, major_idx++) {
    for (int loop_x = 0, tile_out_x = tile_out_x_base;
         loop_x < p.loop_x & tile_out_x < p.out_w;
         loop_x++, tile_out_x += tile_out_w) {
      int tile_mid_x = tile_out_x * down_x + up_x - 1 - p.pad_x0;
      int tile_mid_y = tile_out_y * down_y + up_y - 1 - p.pad_y0;
      int tile_in_x = floor_div(tile_mid_x, up_x);
      int tile_in_y = floor_div(tile_mid_y, up_y);
      __syncthreads();
      for (int in_idx = threadIdx.x; in_idx < tile_in_h * tile_in_w;
           in_idx += blockDim.x) {
        int rel_in_y = in_idx / tile_in_w;
        int rel_in_x = in_idx - rel_in_y * tile_in_w;
        int in_x = rel_in_x + tile_in_x;
        int in_y = rel_in_y + tile_in_y;
        scalar_t v = 0.0;
        if (in_x >= 0 & in_y >= 0 & in_x < p.in_w & in_y < p.in_h) {
          v = input[((major_idx * p.in_h + in_y) * p.in_w + in_x) *
                        p.minor_dim +
                    minor_idx];
        }
        sx[rel_in_y][rel_in_x] = v;
      }
      __syncthreads();
      for (int out_idx = threadIdx.x; out_idx < tile_out_h * tile_out_w;
           out_idx += blockDim.x) {
        int rel_out_y = out_idx / tile_out_w;
        int rel_out_x = out_idx - rel_out_y * tile_out_w;
        int out_x = rel_out_x + tile_out_x;
        int out_y = rel_out_y + tile_out_y;
        int mid_x = tile_mid_x + rel_out_x * down_x;
        int mid_y = tile_mid_y + rel_out_y * down_y;
        int in_x = floor_div(mid_x, up_x);
        int in_y = floor_div(mid_y, up_y);
        int rel_in_x = in_x - tile_in_x;
        int rel_in_y = in_y - tile_in_y;
        int kernel_x = (in_x + 1) * up_x - mid_x - 1;
        int kernel_y = (in_y + 1) * up_y - mid_y - 1;
        scalar_t v = 0.0;
#pragma unroll
        for (int y = 0; y < kernel_h / up_y; y++)
#pragma unroll
          for (int x = 0; x < kernel_w / up_x; x++)
            v += sx[rel_in_y + y][rel_in_x + x] *
                 sk[kernel_y + y * up_y][kernel_x + x * up_x];
        if (out_x < p.out_w & out_y < p.out_h) {
          out[((major_idx * p.out_h + out_y) * p.out_w + out_x) * p.minor_dim +
              minor_idx] = v;
        }
      }
    }
  }
}
"""
src = r"""
  UpFirDn2DKernelParams p;
  
  p.major_dim = in0_shape0;
  p.in_h = in0_shape1;
  p.in_w = in0_shape2;
  p.minor_dim = in0_shape3;
  p.kernel_h = in1_shape0;
  p.kernel_w = in1_shape1;
  p.up_x = in2_shape0;
  p.up_y = in2_shape1;
  p.down_x = in2_shape2;
  p.down_y = in2_shape3;
  p.pad_x0 = in2_shape4;
  p.pad_x1 = in2_shape5;
  p.pad_y0 = in2_shape6;
  p.pad_y1 = in2_shape7;
  p.out_h = (p.in_h * p.up_y + p.pad_y0 + p.pad_y1 - p.kernel_h + p.down_y) / p.down_y;
  p.out_w = (p.in_w * p.up_x + p.pad_x0 + p.pad_x1 - p.kernel_w + p.down_x) / p.down_x;
  int mode = -1;
  int tile_out_h = -1;
  int tile_out_w = -1;
  if (p.up_x == 1 && p.up_y == 1 && p.down_x == 1 && p.down_y == 1 &&
      p.kernel_h <= 4 && p.kernel_w <= 4) {
    mode = 1;
    tile_out_h = 16;
    tile_out_w = 64;
  }
  if (p.up_x == 1 && p.up_y == 1 && p.down_x == 1 && p.down_y == 1 &&
      p.kernel_h <= 3 && p.kernel_w <= 3) {
    mode = 2;
    tile_out_h = 16;
    tile_out_w = 64;
  }
  if (p.up_x == 2 && p.up_y == 2 && p.down_x == 1 && p.down_y == 1 &&
      p.kernel_h <= 4 && p.kernel_w <= 4) {
    mode = 3;
    tile_out_h = 16;
    tile_out_w = 64;
  }
  if (p.up_x == 2 && p.up_y == 2 && p.down_x == 1 && p.down_y == 1 &&
      p.kernel_h <= 2 && p.kernel_w <= 2) {
    mode = 4;
    tile_out_h = 16;
    tile_out_w = 64;
  }
  if (p.up_x == 1 && p.up_y == 1 && p.down_x == 2 && p.down_y == 2 &&
      p.kernel_h <= 4 && p.kernel_w <= 4) {
    mode = 5;
    tile_out_h = 8;
    tile_out_w = 32;
  }
  if (p.up_x == 1 && p.up_y == 1 && p.down_x == 2 && p.down_y == 2 &&
      p.kernel_h <= 2 && p.kernel_w <= 2) {
    mode = 6;
    tile_out_h = 8;
    tile_out_w = 32;
  }
  
  dim3 block_size;
  dim3 grid_size;
  if (tile_out_h > 0 && tile_out_w > 0) {
    p.loop_major = (p.major_dim - 1) / 16384 + 1;
    p.loop_x = 1;
    block_size = dim3(32 * 8, 1, 1);
    grid_size = dim3(((p.out_h - 1) / tile_out_h + 1) * p.minor_dim,
                     (p.out_w - 1) / (p.loop_x * tile_out_w) + 1,
                     (p.major_dim - 1) / p.loop_major + 1);
  } else {
    p.loop_major = (p.major_dim - 1) / 16384 + 1;
    p.loop_x = 4;
    block_size = dim3(4, 32, 1);
    grid_size = dim3((p.out_h * p.minor_dim - 1) / block_size.x + 1,
                     (p.out_w - 1) / (p.loop_x * block_size.y) + 1,
                     (p.major_dim - 1) / p.loop_major + 1);
  }
  
  switch (mode) {
    case 1:
      upfirdn2d_kernel<scalar_t, 1, 1, 1, 1, 4, 4, 16, 64>
          <<<grid_size, block_size>>>((float *)out0_p,
                                                 (const float *)in0_p,
                                                 (const float *)in1_p, p);
      break;
    case 2:
      upfirdn2d_kernel<scalar_t, 1, 1, 1, 1, 3, 3, 16, 64>
          <<<grid_size, block_size>>>((float *)out0_p,
                                                 (const float *)in0_p,
                                                 (const float *)in1_p, p);
      break;
    case 3:
      upfirdn2d_kernel<scalar_t, 2, 2, 1, 1, 4, 4, 16, 64>
          <<<grid_size, block_size>>>((float *)out0_p,
                                                 (const float *)in0_p,
                                                 (const float *)in1_p, p);
      break;
    case 4:
      upfirdn2d_kernel<scalar_t, 2, 2, 1, 1, 2, 2, 16, 64>
          <<<grid_size, block_size>>>((float *)out0_p,
                                                 (const float *)in0_p,
                                                 (const float *)in1_p, p);
      break;
    case 5:
      upfirdn2d_kernel<scalar_t, 1, 1, 2, 2, 4, 4, 8, 32>
          <<<grid_size, block_size>>>((float *)out0_p,
                                                 (const float *)in0_p,
                                                 (const float *)in1_p, p);
      break;
    case 6:
      upfirdn2d_kernel<scalar_t, 1, 1, 2, 2, 4, 4, 8, 32>
          <<<grid_size, block_size>>>((float *)out0_p,
                                                 (const float *)in0_p,
                                                 (const float *)in1_p, p);
      break;
    default:
      upfirdn2d_kernel_large<scalar_t><<<grid_size, block_size>>>(
          (float *)out0_p, (const float *)in0_p,
          (const float *)in1_p, p);
    }
"""


class UpFirDn2dBackward(jt.Function):
    def __init__(self):
        super().__init__()
        self.up_x, self.up_y, self.down_x, self.down_y = None, None, None, None
        self.pad_x0, self.pad_x1, self.pad_y0, self.pad_y1 = None, None, None, None
        self.in_size, self.out_size = None, None

    def execute(self, grad_output, kernel, grad_kernel, up, down, pad, g_pad, in_size, out_size):
        up_x, up_y = up
        down_x, down_y = down
        g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1 = g_pad
        _, _, in_h, in_w = grad_output.shape
        kernel_h, kernel_w = grad_kernel.shape

        grad_output = grad_output.view(-1, out_size[0], out_size[1], 1)

        out_h = (in_h * down_y + g_pad_y0 + g_pad_y1 - kernel_h + up_y) // up_y
        out_w = (in_w * down_x + g_pad_x0 + g_pad_x1 - kernel_w + up_x) // up_x
        grad_input = jt.code(
            [grad_output.shape[0], out_h, out_w, grad_output.shape[3]], grad_output.dtype,
            [grad_output, grad_kernel, jt.empty((down_x, down_y, up_x, up_y, g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1))],
            cuda_header=header, cuda_src=src
        )
        # grad_input.sync()
        # jt.sync_all()
        grad_input = grad_input.view(in_size)

        self.save_vars = kernel

        pad_x0, pad_x1, pad_y0, pad_y1 = pad

        self.up_x, self.up_y, self.down_x, self.down_y = up_x, up_y, down_x, down_y
        self.pad_x0, self.pad_x1, self.pad_y0, self.pad_y1 = pad_x0, pad_x1, pad_y0, pad_y1
        self.in_size, self.out_size = in_size, out_size

        return grad_input

    def grad(self, gradgrad_input):
        kernel = self.save_vars
        _, _, in_h, in_w = gradgrad_input.shape
        kernel_h, kernel_w = kernel.shape

        gradgrad_input = gradgrad_input.view(-1, self.in_size[2], self.in_size[3], 1)
        out_h = (in_h * self.up_y + self.pad_y0 + self.pad_y1 - kernel_h + self.down_y) // self.down_y
        out_w = (in_w * self.up_x + self.pad_x0 + self.pad_x1 - kernel_w + self.down_x) // self.down_x
        gradgrad_out = jt.code(
            [gradgrad_input.shape[0], out_h, out_w, gradgrad_input.shape[3]], gradgrad_input.dtype,
            [gradgrad_input, kernel, jt.empty((self.up_x, self.up_y, self.down_x, self.down_y, self.pad_x0, self.pad_x1, self.pad_y0, self.pad_y1))],
            cuda_header=header, cuda_src=src
        )
        # gradgrad_out.sync()
        # jt.sync_all()
        gradgrad_out = gradgrad_out.view(self.in_size[0], self.in_size[1], self.out_size[0], self.out_size[1])

        return gradgrad_out, None, None, None, None, None, None, None, None


class UpFirDn2d(jt.Function):
    def __init__(self):
        super(UpFirDn2d, self).__init__()
        self.kernel, self.grad_kernel = None, None
        self.in_size, self.out_size, self.up, self.down, self.down = None, None, None, None, None
        self.pad, self.g_pad = None, None

    def execute(self, x, kernel, up, down, pad):
        up_x, up_y = up
        down_x, down_y = down
        pad_x0, pad_x1, pad_y0, pad_y1 = pad

        kernel_h, kernel_w = kernel.shape
        batch, channel, in_h, in_w = x.shape
        self.in_size = x.shape

        x = x.reshape((-1, in_h, in_w, 1))

        self.kernel, self.grad_kernel = kernel, jt.flip(kernel, [0, 1])

        out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
        out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
        self.out_size = (out_h, out_w)

        self.up = (up_x, up_y)
        self.down = (down_x, down_y)
        self.pad = (pad_x0, pad_x1, pad_y0, pad_y1)

        g_pad_x0 = kernel_w - pad_x0 - 1
        g_pad_y0 = kernel_h - pad_y0 - 1
        g_pad_x1 = in_w * up_x - out_w * down_x + pad_x0 - up_x + 1
        g_pad_y1 = in_h * up_y - out_h * down_y + pad_y0 - up_y + 1

        self.g_pad = (g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1)

        out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h + down_y) // down_y
        out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w + down_x) // down_x
        out = jt.code(
            [x.shape[0], out_h, out_w, x.shape[3]], x.dtype,
            [x, kernel, jt.empty((up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1))],
            cuda_header=header, cuda_src=src
        )
        # out.sync()
        # jt.sync_all()

        return out.view(-1, channel, out_h, out_w)

    def grad(self, grad_output):
        kernel, grad_kernel = self.kernel, self.grad_kernel

        grad_input = UpFirDn2dBackward.apply(
            grad_output, kernel, grad_kernel,
            self.up, self.down, self.pad, self.g_pad, self.in_size, self.out_size
        )

        return grad_input, None, None, None, None


def upfirdn2d(x, kernel, up=1, down=1, pad=(0, 0), cuda=True):
    if cuda:
        out = UpFirDn2d.apply(x, kernel, (up, up), (down, down), (pad[0], pad[1], pad[0], pad[1]))
    else:
        out = upfirdn2d_native(
            x, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1]
        )

    return out


def upfirdn2d_native(
    x, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
):
    _, channel, in_h, in_w = x.shape
    x = x.reshape(-1, in_h, in_w, 1)

    _, in_h, in_w, minor = x.shape
    kernel_h, kernel_w = kernel.shape

    out = x.view(-1, in_h, 1, in_w, 1, minor)
    out = nn.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    out = nn.pad(
        out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)]
    )
    out = out[
        :,
        max(-pad_y0, 0): out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0): out.shape[2] - max(-pad_x1, 0),
        :,
    ]

    out = out.permute(0, 3, 1, 2)
    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    )

    w = jt.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = nn.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down_y, ::down_x, :]

    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1

    return out.view(-1, channel, out_h, out_w)
