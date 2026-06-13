# Complex Numbers for AI

## Hook

Complex numbers aren't a math curiosity—they're the representation that makes Fourier transforms, quantum states, and certain neural network architectures computationally tractable. If you've ever used an FFT or seen a spectrogram, you've relied on Euler's formula without knowing it.

## Concept

The complex plane, Euler's formula ($e^{i\theta} = \cos\theta + i\sin\theta$), and why multiplying by a complex number is a rotation and scaling operation. Covers polar vs. rectangular form and conjugates.

## Mechanism

The Discrete Fourier Transform expressed as complex matrix multiplication: why $X[k] = \sum x[n] \cdot e^{-i2\pi kn/N}$ maps time-domain signals to frequency-domain coefficients. The magnitude gives signal strength; the phase gives timing. This is the mechanism behind spectrograms, MFCCs, and frequency-domain convolution.

## Code

Working Python implementation: compute DFT by hand with complex arithmetic, then compare against `numpy.fft.fft`. Print magnitudes and phases to confirm both produce identical output. Second example: demonstrate that complex multiplication is rotation by multiplying a signal vector by $e^{i\pi/4}$ and printing before/after.

## Use It

Fourier analysis underpins audio feature extraction in conversational intelligence platforms (Gong, Chorus). The MFCC pipeline that powers sales-call transcription and speaker diarization starts with a DFT of complex-valued coefficients. This is foundational for Zone 2 (Processing) in the GTM map—specifically the signal-processing layer that feeds intent classification.

## Ship It

- **Easy**: Write a function that converts rectangular complex numbers to polar form and back; verify round-trip fidelity with assert statements.
- **Medium**: Implement a 1D DFT using only complex multiplication (no FFT call), validate output against `numpy.fft.fft` on a test signal of length 64.
- **Hard**: Compute and print the spectrogram of a generated chirp signal (frequency increasing over time), then identify the single frequency bin with highest magnitude at each time step.