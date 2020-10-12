import "@tensorflow/tfjs-node";

type RandomMath = Math & { readonly seedrandom: (seed: number) => string };

export const seed = 1234567890;

(Math as RandomMath).seedrandom(seed);
