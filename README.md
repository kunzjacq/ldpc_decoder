# OpenCL LDPC decoder

A Low-Density Parity-Check error-correcting code decoder implemented in C++14 and OpenCL, tuned for large codes close to the Shannon bound. 
Codes are read in the `alist` text format; no code generator is included.
Typical decoding speed on semi-recent discrete graphics cards is a between 10 and 100Mb/s, for codes with codewords of size 10<sup>6</sup>. 

Decoding is performed with the flood soft-decoding algorithm. This is an iterative algorithm which requires a variable number of rounds to complete. There can be large variations of the number of rounds required to decode a frame, even at a fixed noise level. Hence a parallel implementation of flood decoding with a fixed number of rounds on a GPU for all frames is inefficient: it needs a large number of rounds to ensure that all frames are properly error-corrected, but most frames are error-free after a much smaller number of rounds and waste computing resources. 

To avoid this inefficiency, the implemented decoder is able to replace on the fly frames that have finished decoding with new frames to decode. To identify finished frames, once every *k* rounds of the flood decoding algorithm, all parity equations of the frames being decoding are computed. Finished frames are the ones whose parity equations are all satisfied. This computation is done on the GPU.

The decoder is not restricted to codewords; instead, it takes as input the target values of the parity equations on the data to decode. This simplifies the testing of the decoder, as random data does not need to be transformed into codewords. The decoder could easily be transformed into a codeword-decoding algorithm by setting all parity bits to 0.

### Noise models

The decoder is able to handle BSC and AWGN channels. Alternatively, it could be used (with modifications) to handle any channel with an input already converted to Log-likelihood Ratios (LLRs).

## Build

The decoder is built with Cmake.

For single-configuration generators, such as make and variants, one does (assuming *sh*-like syntax):

    cmake -S $SOURCE_DIR -B $BUILD_DIR -D CMAKE_BUILD_TYPE=$CONFIG
    cmake --build $BUILD_DIR 

For multi-configuration generators, e.g. when targeting Visual Studio under Windows, one does (assuming windows CMD syntax):

    cmake -S %SOURCE_DIR% -B %BUILD_DIR%
    cmake --build %BUILD_DIR% --config %CONFIG%

with `CONFIG` equal to `Debug` or `Release`. 

There is only one target, `ldpc_decoder_gpu`, which implements a testing program for the decoder, that generates data to decode, and measures the result of the decoding process and exection times. A usage example is given below. 

Visual Studio, a linux gcc or clang, or MinGW gcc under Windows can be used to build the project, with some caveats. 

The project is dependent on finding a working implementation of OpenCL. CMake is able to find such an implementation if it is present, under Windows and Linux, but it won't find a Windows OpenCL implementation from MSYS or Cygwin because of conversion issues between unix-like and Windows paths. The best way to build the project using MinGW gcc is therefore to use a Windows CMD shell. There is some limited support to use the Nvidia CUDA OpenCL implementation from within MSYS, but it is probably not very reliable.

### AVX2 Issues with MinGW and Visual studio

The source code uses AVX2 for some auxiliary functions. Unfortunately, this causes problems with the compilers listed below.
  * MinGW gcc under Windows, in any version, is not able to properly align 256-bit AVX2 variables on the stack (to a 32-byte alignment); the cases that would make the program segfault have been avoided in the implementation seemingly with some success, but problems may very well remain until non-AVX2 versions of the problematic functions are implemented. See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=54412. 

  * Visual Studio before version 16.3 also miscompiles some AVX2 code. See https://developercommunity.visualstudio.com/t/avx2-optimization-bug-with-cl-versions-1916270301/549774.

## Usage example

The program must have its OpenCL kernels in a subdirectory `src/opencl/` relative to its execution path. It should therefore be started from the root of the repository.

`ldpc_decoder_gpu -f code_awgn_rate_0.5_thr_0.95.alist -c  1 -n 0.94 -p 8 -m 2 -e 15 -i 120`

  * `-f code.alist`: load code file `code.alist`
  * `-c  1`: test it with channel type 1 (AWGN)
  * `-n 0.94`: use noise level 0.94. For AWGN channels, the noise level is the stdev of the gaussian noise.
  * `-p 4`: ask to decode n = 2<sup>8</sup> = 256 frames in parallel on the GPU. This number may be lowered by the decoder as it is limited by the available memory on the GPU.
  * `-m 2`: use a loading factor of 2, i.e. generate 2×n = 512 frames. Frames are processed in order starting with the first n ones and new frames are sent to the GPU when previous frames have been error-corrected. Higher loading factors ensure the GPU is kept busy during a larger fraction of the test and results in better overall throughput.
  * `-e 15`: consider frames with less than 15 errors as corrected when computing final Frame Error Rate statistics. 
  * `-i 120`: run at most 120 iterations of the decoding algorithm per frame. Frames that are not fully error-corrected after this amount of iterations will be retired from the GPU anyway. Without this option, the default value for this parameter is 100.

The full list of options can be obtained with `ldpc_decoder_gpu.exe -h`.

With such parameters and one of the sample codes available below (with frame size 2<sup>20</sup>), the output of this test is 

                                                ***
                                              Summary

    * Channel and code description

    Channel:
    Binary channel with Gaussian noise of std. deviation 0.94; SNR = 1.13173
    capacity: 0.526758 bits/symbol

    Error-correcting code:
    1048576 variables
    611669 parity bits
    174763 erased variables (not sent, but recovered)
    Rate = 0.500001

    Code efficiency over channel = rate/channel capacity = 94.92%


    * Test result

    # of frames decoded:              512
    Frame size:                       1048576 bits
    Total # of errors:                156
    Bit error rate (BER):             2.90573e-07
    Maximum # of errors / frame:      67
    Frames with more than 15 errors:  4 (corresponding FER: 0.0078125)
    Frames with at least one error:   7 (corresponding FER: 0.0136719)

    Mbits processed:                  512
    Elapsed system time:              11.2469 sec.
    throughput:                       45.5235 Mbits/sec.

## sample codes

Two codes with codeword size of 2<sup>20</sup> = 1,048,576 are present in the repository to test the decoder.

  * `code_awgn_rate_0.5_thr_0.95.alist` can correct gaussian noise up to std dev of 0.95.
  * `code_bsc_rate_0.9_thr_0.09.alist` can correct a binary symmetric noise of error probability up to 0.09.
