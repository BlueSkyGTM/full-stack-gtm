# Voice Activity Detection & Turn-Taking — Silero, Cobra, and the Flush Trick

## Learning Objectives

- Implement frame-by-frame VAD inference using Silero's neural classifier on raw audio chunks
- Compare speech detection behavior between Silero and Cobra across identical audio samples
- Apply the flush trick to recover speech from residual VAD buffer data at stream teardown
- Configure silence threshold and minimum speech duration parameters for turn-taking in voice agent deployments
- Measure missed-speech rate differences between flushed and unflushed buffer handling on a streaming simulation

## The Problem

Voice agents make three distinct decisions on every audio chunk they receive. First: is this frame speech or silence? Second: has the user started a new utterance? Third: have they finished talking? Get any of these wrong and the conversation breaks. Interrupt too early and you cut off the prospect mid