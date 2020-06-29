import { createNetwork, prepareData, train } from "./";
import carsData from "./data/cars.json";

describe("basic network", () => {
	it("trains", async () => {
		const model = createNetwork();
		const { inputs, labels } = prepareData(carsData);
		const history = await train(model, inputs, labels);
		const losses = history.history.loss;
		return expect(losses[losses.length - 1]).toBeLessThan(0.05);
	});
});
