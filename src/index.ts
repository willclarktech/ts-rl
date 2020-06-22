import * as tf from "@tensorflow/tfjs-node";

export const createNetwork = (): tf.Sequential => {
	const network = tf.sequential();
	network.add(
		tf.layers.dense({
			inputShape: [1],
			units: 1,
		}),
	);
	network.add(
		tf.layers.dense({
			units: 1,
		}),
	);
	return network;
};
