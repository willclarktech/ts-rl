import { CartPole } from "./cart-pole";
import { Environment } from "./core";
import { ReinforceLearner } from "./reinforce";
import { logEpisode, mean } from "./util";

const train = (
	env: Environment,
	learner: ReinforceLearner,
	maxEpisodes: number,
	rollingAveragePeriod: number,
	logPeriod: number,
	logFile: string,
): boolean => {
	const returns = [];
	const rollingAverageReturns = [];

	for (let episode = 1; episode <= maxEpisodes; ++episode) {
		const ret = learner.runEpisode(env);
		returns.push(ret);

		const rollingAverageReturn = mean(returns.slice(-100));
		rollingAverageReturns.push(rollingAverageReturn);

		if (episode % logPeriod === 0) {
			logEpisode(
				episode,
				returns,
				rollingAveragePeriod,
				rollingAverageReturns,
				logFile,
			);
		}

		if (
			env.winningScore !== undefined &&
			episode >= rollingAveragePeriod &&
			rollingAverageReturn >= env.winningScore
		) {
			return true;
		}
	}

	return false;
};

const main = (): void => {
	const env = new CartPole();

	const hiddenWidths = [4, 4];
	const alpha = 0.001; // Learning rate
	const gamma = 0.99; // Discount rate
	const learner = new ReinforceLearner(env, hiddenWidths, alpha, gamma);

	const maxEpisodes = 1000;
	const rollingAveragePeriod = 100;
	const logPeriod = 100;
	const experimentName = `${env.name}-${learner.name}`;
	const logFile = `./results/data/${experimentName}.json`;

	console.info("Score to beat:", env.winningScore);

	const didWin = train(
		env,
		learner,
		maxEpisodes,
		rollingAveragePeriod,
		logPeriod,
		logFile,
	);

	console.info(didWin ? "You won!" : "You lost.");
};

main();
