import carsData from "./data/cars.json";
import { prepareData, train } from "./linear-regression";
import { createNetwork } from "./util";

describe("basic network", () => {
	it("trains", async () => {
		const { normalizedInputs, normalizedLabels } = prepareData(carsData);

		const activationFunction = "sigmoid";
		const model = createNetwork([1, 4, 4, 1], activationFunction);

		const hyperparameters = {
			batchSize: 32,
			epochs: 20,
			shuffle: true,
			learningRate: 0.03,
		};
		const history = await train(
			model,
			normalizedInputs,
			normalizedLabels,
			hyperparameters,
		);

		const losses = history.history.loss;
		return expect(losses[losses.length - 1]).toBeLessThan(0.05);
	});
});
