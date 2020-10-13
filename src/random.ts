import "@tensorflow/tfjs-node";

import { seed } from "./options";

type RandomMath = Math & { readonly seedrandom: (seed: number) => string };

(Math as RandomMath).seedrandom(seed);
