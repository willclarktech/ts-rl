import { Agent, Random, Reinforce } from "./agents";
import { Blackjack, CartPole, Environment } from "./environments";
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

const envs: { readonly [key: string]: () => Environment } = {
	blackjack: (): Environment => new Blackjack(),
	cartpole: (): Environment => new CartPole(),
};

const createAgent = (agentName: string, env: Environment): Agent => {
	switch (agentName) {
		case "random": {
			return new Random(env);
		}
		case "reinforce": {
			const hiddenWidths = [8];
			const alpha = 0.001; // Learning rate
			const gamma = 0.99; // Discount rate
			return new Reinforce(env, hiddenWidths, alpha, gamma);
		}
		default:
			throw new Error("Agent name not recognised");
	}
};

const main = (): void => {
	const environmentName = process.argv[2] ?? "cartpole";
	const agentName = process.argv[3] ?? "reinforce";

	const createEnv: () => Environment = envs[environmentName];
	if (createEnv === undefined) {
		throw new Error("Environment name not recognised");
	}
	const env = createEnv();
	const agent = createAgent(agentName, env);

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

// Run with eg `npm start [Blackjack]`
main();
