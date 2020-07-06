import * as tf from "@tensorflow/tfjs-node";

type ActivationFunction = "sigmoid" | "relu";

export const createNetwork = (
	widths: readonly number[],
	activation: ActivationFunction,
): tf.Sequential => {
	const network = tf.sequential({
		name: "tutorial-2d",
		layers: widths.slice(1).map((width, i) =>
			i === 0
				? tf.layers.dense({
						inputShape: [widths[0]],
						units: width,
						activation,
				  })
				: i !== widths.length - 2
				? tf.layers.dense({ units: width, activation })
				: tf.layers.dense({ units: width }),
		),
	});
	return network;
};
