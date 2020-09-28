import { Agent, Reinforce } from "./agents";
import { CartPole, Environment } from "./environments";
import { logEpisode, mean } from "./util";

const train = (
	env: Environment,
	agent: Agent,
	maxEpisodes: number,
	rollingAveragePeriod: number,
	logPeriod: number,
	logFile: string,
): boolean => {
	const returns = [];
	const rollingAverageReturns = [];

	for (let episode = 1; episode <= maxEpisodes; ++episode) {
		const ret = agent.runEpisode(env);
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
	const env: Environment = new CartPole();

	const hiddenWidths = [4, 4];
	const alpha = 0.001; // Learning rate
	const gamma = 0.99; // Discount rate
	const agent: Agent = new Reinforce(env, hiddenWidths, alpha, gamma);

	const maxEpisodes = 1000;
	const rollingAveragePeriod = 100;
	const logPeriod = 100;
	const experimentName = `${env.name}-${agent.name}`;
	const logFile = `./results/data/${experimentName}.json`;

	console.info("Score to beat:", env.winningScore ?? "[not set]");

	const didWin = train(
		env,
		agent,
		maxEpisodes,
		rollingAveragePeriod,
		logPeriod,
		logFile,
	);

	console.info(didWin ? "You won!" : "You lost.");
};

main();
