# pixelhasher: GPU miner for zkBits

First open-source release of pixelhasher for Linux/Windows.

Source is being ported over from closed-source version of pixelhasher by
rockmtn. Full source will be available soon.(tm)

## Closed-source beta

The closed-source version is available for download from the official zkBits
website.

Note: Pixelhasher must be connected to the [solo mining helper](https://github.com/zkbits/pixelhasher-solo-helper), or to the mining
pool (coming soon).

- Latest release: Pixelhasher v0.1.0 (`pixelhasher_v0.1.0.zip`)
- Solo mining adapter: [zkbits/pixelhasher-solo-helper](https://github.com/zkbits/pixelhasher-solo-helper) by 0xBrian

Linux install instructions:

```
wget https://zkbits.letz.dev/release/pixelhasher_v0.1.0.zip
unzip pixelhasher_v0.1.0.zip
cd pixelhasher_v0.1.0/
./pixelhasher
```

# How fast will my card mine?

| pixelhasher version | tester | card | hashrate gh/s | relmax |
| --- | --- | --- | --- | --- |
| v0.1.0 | rockmtn | RTX 4090 | 4.64 GH/s | 100% |
| v0.1.0 | rockmtn | L40S | 4.02 GH/s | 87% |
| v0.1.0 | rockmtn | RTX 6000 Ada Generation | 3.57 GH/s | 77% |
| v0.1.0 | rockmtn | RTX 4080 Super | 2.21 GH/s | 48% |
| v0.1.0 | rockmtn | RTX 4070 Ti Super | 2.19 GH/s | 47% |
| v0.1.0 | rockmtn | RTX 3090 Ti | 1.65 GH/s | 36% |
| v0.1.0 | rockmtn | A100-SXM4-40GB | 1.62 GH/s | 35% |
| v0.1.0 | rockmtn | RTX 3080 | 1.59 GH/s | 34% |
| v0.1.0 | rockmtn | RTX 3080 Ti | 1.53 GH/s | 33% |
| v0.1.0 | rockmtn | RTX A6000 | 1.52 GH/s | 33% |
| v0.1.0 | rockmtn | RTX 3090 | 1.50 GH/s | 32% |
| v0.1.0 | rockmtn | RTX 2080 Ti | 1.49 GH/s | 32% |
| v0.1.0 | rockmtn | A40 | 1.45 GH/s | 31% |
| v0.1.0 | rockmtn | Quadro RTX 6000 | 1.41 GH/s | 30% |
| v0.1.0 | rockmtn | RTX A5000 | 1.35 GH/s | 29% |
| v0.1.0 | d | Tesla V100-SXM2-16GB | 1.30 GH/s | 28% |
| v0.1.0 | rockmtn | TITAN V | 1.15 GH/s | 25% |
| v0.1.0 | rockmtn | RTX A4500 | 1.14 GH/s | 24% |
| v0.1.0 | rockmtn | RTX 4060 Ti | 1.12 GH/s | 24% |
| v0.1.0 | rockmtn | RTX 3070 | 1.12 GH/s | 24% |
| v0.1.0 | U | 1080 Ti | 1.06 GH/s | 23% |
| v0.1.0 | rockmtn | RTX A4000 | 0.91 GH/s | 20% |
| v0.1.0 | rockmtn | RTX 2070S | 0.87 GH/s | 19% |
| v0.1.0 | rockmtn | RTX 3060 LHR | 0.65 GH/s | 14% |
| v0.1.0 | rockmtn | RTX 3060 | 0.63 GH/s | 14% |
| v0.1.0 | rockmtn | GTX 1070 | 0.62 GH/s | 13% |
| v0.1.0 | rockmtn | GTX 1660 Ti | 0.56 GH/s | 12% |
| v0.1.0 | rockmtn | RTX A2000 | 0.50 GH/s | 11% |
