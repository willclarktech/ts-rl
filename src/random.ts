import "@tensorflow/tfjs-node";

type RandomMath = Math & { readonly seedrandom: (seed: number) => string };

export const setSeed = (seed: number): void => {
	(Math as RandomMath).seedrandom(seed);
};
