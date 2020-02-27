#!/bin/bash

for i in {15..32}
do
    mkdir ./renderings/midi/wolfram_eca_8bit_seed_${i}bit
    python midi.py --mode=wolfram --outdir ./renderings/midi/wolfram_eca_8bit_seed_${i}bit --numSeedBits=$i
done

# Test all

for i in {15..32}
do
    python midi.py --test=renderings/midi/wolfram_eca_8bit_seed_${i}bit/*_states.json
done